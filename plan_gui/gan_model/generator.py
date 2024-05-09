import torch
import torch.nn as nn
from torch.nn import Parameter

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from concurrent.futures import ThreadPoolExecutor

from pyctm.representation.sdr_idea_array import SDRIdeaArray

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerXLAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super(TransformerXLAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.dropout = dropout

        # Initialize query, key, and value linear transformations
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        # Initialize output linear transformation
        self.W_O = nn.Linear(d_model, d_model)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        # Linear transformations
        Q = self.W_Q(Q)
        K = self.W_K(K)
        V = self.W_V(V)

        # Split heads
        Q = self.split_heads(Q, self.n_head)
        K = self.split_heads(K, self.n_head)
        V = self.split_heads(V, self.n_head)

        # Scale dot product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)

        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Weighted sum of values
        attn_output = torch.matmul(attn_probs, V)

        # Combine heads
        attn_output = self.combine_heads(attn_output)

        # Linear transformation for output
        attn_output = self.W_O(attn_output)

        return attn_output

    def split_heads(self, x, n_head):
        batch_size, seq_len, d_model = x.size()
        head_dim = d_model // n_head
        x = x.view(batch_size, seq_len, n_head, head_dim)
        return x.transpose(1, 2)

    def combine_heads(self, x):
        batch_size, n_head, seq_len, head_dim = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, n_head * head_dim)


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class ResidualLayer(nn.Module):
    def __init__(self, sublayer, input_dim):
        super(ResidualLayer, self).__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        return x + self.sublayer(self.norm(x))

class ConvolutionalEmbeddingLayer1D(nn.Module):
    def __init__(self, input_dim, d_model):
        super(ConvolutionalEmbeddingLayer1D, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x.permute(0, 2, 1)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.permute(0, 2, 1)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = TransformerXLAttention(d_model, n_head, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = TransformerXLAttention(d_model, n_head, dropout)
        self.cross_attn = TransformerXLAttention(d_model, n_head, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x


class PlanningTransformer(nn.Module):
    def __init__(self, vocabulary_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, max_seq_len=20, device='cpu'):
        super(PlanningTransformer, self).__init__()

        self.vocabulary_size = vocabulary_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.device = device

        self.encoder_embedding = nn.Embedding(vocabulary_size, d_model)
        self.decoder_embedding = nn.Embedding(vocabulary_size, d_model)

        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_encoder_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_decoder_layers)])

        self.dropout = nn.Dropout(dropout)

        self.output_layer =nn.Linear(d_model, self.vocabulary_size)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool().to(self.device)
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def create_positional_encoding(self, max_len, d_model):
        # Create a matrix of positional encodings
        positional_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0)
        return nn.Parameter(positional_encoding, requires_grad=False)

    def forward(self, src, tgt):
        batch_size_tgt, seq_len_tgt = tgt.size()
        batch_size, seq_len = src.size()

        src_mask, tgt_mask = self.generate_mask(src, tgt)

        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.output_layer(dec_output)

        output = output.view(batch_size_tgt, seq_len_tgt, self.vocabulary_size)
        return output
    

# def beam_search(model, src, start_symbol, end_symbol, max_len, beam_size, temperature, device):
#     src = src.to(device)
#     ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data).to(device)
#     beam = [(ys, 0.0)]
#     finished_beams = []

#     for _ in range(max_len - 1):
#         candidates = []
#         for ys, score in beam:
#             if ys[-1, -1].eq(end_symbol).item():
#                 finished_beams.append((ys, score))
#                 continue

#             with torch.no_grad():
#                 logits = model(src, ys)[-1, :]
#             logits = logits / temperature
#             probs = F.softmax(logits, dim=-1)
#             top_probs, top_ix = probs.topk(beam_size)

#             for i in range(beam_size):
#                 prob = top_probs[-1, i].item()
#                 ix = top_ix[-1, i].item()
#                 next_ys = torch.cat([ys, torch.tensor([[ix]], device=device)], dim=1)
#                 next_score = score - math.log(prob)
#                 candidates.append((next_ys, next_score))

#         beam = sorted(candidates, key=lambda x: x[1])[:beam_size]

#     if not finished_beams:
#         finished_beams = beam

#     ys, _ = sorted(finished_beams, key=lambda x: x[1])[-1]
#     return ys
    
def get_graph_connection():
    graph_connection = {
        "1.0": ["2.0", "16.0"],
        "2.0": ["1.0", "3.0", "15.0"],
        "3.0": ["2.0", "4.0", "14.0"],
        "4.0": ["3.0", "5.0"],
        "5.0": ["4.0", "6.0", "14.0"],
        "6.0": ["5.0", "7.0", "13.0"],
        "7.0": ["6.0", "8.0"],
        "8.0": ["7.0", "9.0", "12.0"],
        "9.0": ["8.0", "10.0"],
        "10.0": ["9.0", "11.0", "16.0"],
        "11.0": ["10.0", "12.0", "15.0"],
        "12.0": ["11.0", "13.0", "8.0"],
        "13.0": ["6.0", "12.0", "14.0"],
        "14.0": ["3.0", "5.0", "13.0", "15.0"],
        "15.0": ["2.0", "11.0", "14.0", "16.0"],
        "16.0": ["1.0", "10.0", "15.0"]
    }

    return graph_connection    

def is_valid_plan(action, initial_node, closest_nodes_from_goal_tag, plan_steps, occupiedNodes=[]):
    if action == 'PICK':
        current_step = plan_steps[0]
        if current_step.name == "moveToNode":
            if float(current_step.value) != float(initial_node):
                current_node = str(current_step.value)
                initial_node = str(initial_node)
                if current_node not in get_graph_connection()[initial_node]:
                    return False

    for i in range(len(plan_steps) - 1):
        current_step = plan_steps[i]
        next_step = plan_steps[i + 1]

        if current_step.name == "moveToNode" and next_step.name == "moveToNode":
            try:
                current_node = str(current_step.value)
                next_node = str(next_step.value)
                if next_node not in get_graph_connection()[current_node]:
                    return False
                
                if current_step.value in occupiedNodes:
                    return False
            except Exception as e:
                return False
            
    return True  
    #last_move_to_node_index = None
    #for i, step in enumerate(plan_steps):
    #    if step.name == "moveToNode":
    #        last_move_to_node_index = i

    #if last_move_to_node_index is not None:
    #    try:
    #        last_node = str(plan_steps[last_move_to_node_index].value)
    #        nodes = [closest_node[0] for closest_node in closest_nodes_from_goal_tag]

    #        if int(float(last_node)) not in nodes:
    #            return False
    #    except Exception as e:
    #        return False          

      


def convert_to_idea_array(tensor, sdr_idea_deserializer):
    sdr_idea = tensor.squeeze(0).detach().cpu().numpy().tolist()

    plan_idea_array = SDRIdeaArray(10, 7, 0)
    plan_idea_array.sdr = sdr_idea

    action_step_idea = sdr_idea_deserializer.deserialize(plan_idea_array)

    full_goal = [action_step_idea]
    for i in range(len(action_step_idea.child_ideas)):
        full_goal.append(action_step_idea.child_ideas[i])
    
    return full_goal

def is_valid_sequence(idea_array):
    for idea in idea_array:
        if not is_valid_idea(idea):
            return False
    return True

    
def is_valid_idea(stepIdea):
    if stepIdea.name == "pick" or stepIdea.name == "place":
        if (isinstance(stepIdea.value, list) and
            len(stepIdea.value) == 2 and
            0 <= stepIdea.value[0] <= 187 and
            1 <= stepIdea.value[1] <= 4):
            return True
        else:
            return False

    elif stepIdea.name == "moveTo":
        if isinstance(stepIdea.value, float) and 0 <= stepIdea.value <= 187:
            return True
        else:
            return False

    elif stepIdea.name == "moveToNode":
        if isinstance(stepIdea.value, float) and 1 <= stepIdea.value <= 16:
            return True
        else:
            return False


    return True
    
# def beam_search(model, src, start_symbol, end_symbol, max_len, beam_size, temperature, sdr_idea_deserializer, device):
#     src = src.to(device)
#     ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data).to(device)
#     beam = [(ys, 0.0)]
#     finished_beams = []

#     for _ in range(max_len - 1):
#         candidates = []
#         for ys, score in beam:
#             if ys[-1, -1].eq(end_symbol).item():
#                 finished_beams.append((ys, score))
#                 continue

#             with torch.no_grad():
#                 logits = model(src, ys)[-1, :]
#             logits = logits / temperature
#             probs = F.softmax(logits, dim=-1)
#             top_probs, top_ix = probs.topk(beam_size)

#             for i in range(beam_size):
#                 prob = top_probs[-1, i].item()
#                 ix = top_ix[-1, i].item()
#                 next_ys = torch.cat([ys, torch.tensor([[ix]], device=device)], dim=1)
#                 next_score = score - math.log(prob)
#                 candidates.append((next_ys, next_score))

#         beam = sorted(candidates, key=lambda x: x[1])[:beam_size]

#     print("Amount of Plans Finished:" + str(len(finished_beams)))

#     if not finished_beams:
#         finished_beams = beam

#     print("Amount of Plans Available:" + str(len(finished_beams)))

#     valid_sequences = []
#     index = 0
#     for ys, score in finished_beams:
#         try:
#             idea_array = convert_to_idea_array(ys, sdr_idea_deserializer)

#             if is_valid_plan(idea_array) and is_valid_sequence(idea_array):
#                 valid_sequences.append((ys, score))

#                 print("\nPlan Index:" + str(index))
            
#                 for i in range(len(idea_array)):
#                     print(f'{i} - {idea_array[i].name} - {idea_array[i].value}')
                
#                 print("Total of Steps:" + str(len(idea_array)))
                
#                 print("Score:" + str(score))

#         except Exception as e:
#                 #print("Not valid Plan!")
#                 1 + 1
#                 # print("Plan:" + ys)
#                 # print("Score:" + str(score))
        
#         index+=1

#     if not valid_sequences:
#         # If no valid sequences found, return the best scoring sequence among the finished beams
#         ys, _ = sorted(finished_beams, key=lambda x: x[1])[-1]
#     else:
#         # Choose the sequence with the highest score among the valid sequences
#         ys, _ = sorted(valid_sequences, key=lambda x: x[1])[-1]

#     return ys

def process_beam_item_monotonic(item, src, model, temperature, beam_size, end_symbol, device):
    ys, score = item
    if ys[-1, -1].eq(end_symbol).item():
        return ys, score, True

    with torch.no_grad():
        logits = model(src, ys)[-1, :]
    logits.div_(temperature) 
    probs = F.softmax(logits, dim=-1)
    
    # Adjusting probabilities to enforce monotonicity
    mask = torch.triu(torch.ones_like(probs), diagonal=1)
    masked_probs = probs * mask + (1 - mask) * torch.min(probs)  # masking lower triangle, replacing with minimum probability
    top_probs, top_ix = masked_probs.topk(beam_size)

    candidates = []
    for i in range(beam_size):
        prob = top_probs[-1, i].item()
        ix = top_ix[-1, i].item()
        next_ys = torch.cat([ys, torch.tensor([[ix]], device=device)], dim=1)
        next_score = score - math.log(prob)
        candidates.append((next_ys, next_score))
    return candidates, score, False


def process_beam_item(item, src, model, temperature, beam_size, end_symbol, device):
    ys, score, _ = item
    if ys[-1, -1].eq(end_symbol).item():
        return ys, score, True, ys.shape[1]

    with torch.no_grad():
        logits = model(src, ys)[-1, :]
    logits.div_(temperature) 
    probs = F.softmax(logits, dim=-1)
    top_probs, top_ix = probs.topk(beam_size)

    candidates = []
    for i in range(beam_size):
        prob = top_probs[-1, i].item()
        ix = top_ix[-1, i].item()
        next_ys = torch.cat([ys, torch.tensor([[ix]], device=device)], dim=1)
        next_score = score - math.log(prob)
        candidates.append((next_ys, next_score, next_ys.shape[1]))
    return candidates, score, False, next_ys.shape[1]

def process_beam_item_with_state(item, src, model, temperature, beam_size, end_symbol, device, sequence):
    ys, score, _ = item
    if ys[-1, -1].eq(end_symbol).item():
        return ys, score, True, ys.shape[1]

    step = None
    if len(sequence) > 0:
        step = sequence.pop(0)
        allowed_tokens = load_index()[step]
    else:
        allowed_tokens = None 

    with torch.no_grad():
        logits = model(src, ys)[-1, :]
    logits.div_(temperature) 
    probs = F.softmax(logits, dim=-1)

    if allowed_tokens is not None:
        probs[~allowed_tokens] = 0.0

    top_limit = min(beam_size, len(load_index()[step]))
    top_probs, top_ix = probs.topk(top_limit)

    candidates = []
    for i in range(top_limit):
        prob = top_probs[-1, i].item()
        ix = top_ix[-1, i].item()
        next_ys = torch.cat([ys, torch.tensor([[ix]], device=device)], dim=1)
        next_score = score - math.log(prob)
        candidates.append((next_ys, next_score, next_ys.shape[1]))
    
    return candidates, score, False, next_ys.shape[1]

def load_full_steps_sequence():
    full_steps_sequence = []
    for step in load_state_sequence():
        full_steps_sequence.append(load_steps()[step])

    return full_steps_sequence


def load_state_sequence(is_parent=False):
    sequence = ["ID", "NAME", "TYPE", "METADATA", "LENGTH", "SPECIAL"]
    if is_parent:
        sequence.insert(0, "PARENT_ID")

    return sequence

def load_steps():
    return {
        "PARENT_ID": ["NUMBER", "NUMBER", "NUMBER", "SIGNAL", "NUMBER", "SIGNAL"],
        "ID": ["NUMBER", "NUMBER", "NUMBER", "SIGNAL", "NUMBER", "SIGNAL"],
        "NAME": ["STRING"],
        "TYPE": ["STRING"],
        "METADATA": ["STRING"],
        "LENGTH": ["NUMBER", "NUMBER", "NUMBER", "SIGNAL", "NUMBER", "SIGNAL"],
        "NUM_VALUE": ["NUMBER", "NUMBER", "NUMBER", "SIGNAL", "NUMBER", "SIGNAL"],
        "STRING_VALUE": ["STRING"],
        "SPECIAL": ["SPECIAL"]
    }

def load_index():
    states = {
        "NUMBER": [6,7,8,9,10,11,12,13,14,15],
        "SIGNAL": [4,5],
        "STRING": [16,19,20,22,23,24,25,27,28,29,30,31,32],
        "TYPE": [17],
        "METADATA": [17, 18, 21, 26],
        "SPECIAL": [0, 1, 2, 3]
    }

    return states;

# def process_beam_item(item, src, model, temperature, beam_size, end_symbol, device):
#     ys, score = item
#     if ys[-1].eq(end_symbol).item():
#         return ys, score, True

#     with torch.no_grad():
#         logits = model(src, ys.unsqueeze(0))[-1, :]
#     logits.div_(temperature) 
#     probs = F.softmax(logits, dim=-1)
#     top_probs, top_ix = probs.topk(beam_size)

#         # Avoiding item calls by working directly with tensors
#     top_log_probs = -torch.log(top_probs[-1, :beam_size]).view(-1, 1)  # Convert to column vector
#     scores = score + top_log_probs  # Update scores in a batched manner

#     # Pre-allocate tensor for new candidate sequences
#     new_ys = torch.cat([ys.repeat(beam_size, 1), top_ix[-1, :beam_size].view(-1, 1)], dim=1)

#     # Convert to list of tuples for compatibility with the rest of the code
#     candidates = list(zip(new_ys, scores.view(-1).tolist()))

#     return candidates, score, False

    # candidates = []
    # for i in range(beam_size):
    #     prob = top_probs[-1, i].item()
    #     ix = top_ix[-1, i].item()
    #     next_ys = torch.cat([ys, torch.tensor([[ix]], device=device)], dim=1)
    #     next_score = score - math.log(prob)
    #     candidates.append((next_ys, next_score))
    # return candidates, score, False

# def beam_search(model, src, start_symbol, end_symbol, max_len, beam_size, temperature, sdr_idea_deserializer, device, occupiedNodes, initial_node, closest_nodes_from_goal_tag, action, prune_factor=0.5):
#     src = src.to(device)
#     ys = torch.ones(1).fill_(start_symbol).type_as(src.data).to(device)
#     beam = [(ys, 0.0)]
#     finished_beams = []

#     for i in range(max_len - 1):
#         beam_candidates = []
#         for item_beam in beam:
#             item_genereted, score, finished = process_beam_item(item_beam, src, model, temperature, beam_size, end_symbol, device)
            
#             if finished:
#                 finished_beams.append((item_genereted, score))
#             else:
#                 beam_candidates.extend(item_genereted)
        
#         beam_candidates.sort(key=lambda x: x[1])
#         beam = beam_candidates[:beam_size]
#         print("Index:" + str(i))

#     if not finished_beams:
#         finished_beams = beam

#     valid_sequences = select_possible_plans(sdr_idea_deserializer, occupiedNodes, initial_node, closest_nodes_from_goal_tag, action, finished_beams)    

#     if not valid_sequences:
#         ys, _ = sorted(finished_beams, key=lambda x: x[1])[-1]
#     else:
#         ys, _ = sorted(valid_sequences, key=lambda x: x[1])[-1]

#     return ys


def beam_search(model, src, start_symbol, end_symbol, max_len, beam_size, temperature, sdr_idea_deserializer, device, occupiedNodes, initial_node, closest_nodes_from_goal_tag, action, prune_factor=0.5):
    src = src.to(device)
    # ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data).to(device)
    ys = torch.LongTensor([[1,  6,  6,  6,  4,  6,  4]]).type_as(src.data)
    beam = [(ys, 0.0, 7.0, load_full_steps_sequence())]
    finished_beams = []

    for i in range(max_len - 6):

        beam_candidates = []
        with ThreadPoolExecutor(max_workers=beam_size) as executor:
            for item, score, finished, size, sequence in executor.map(
                lambda item: process_beam_item_with_state(item, src, model, temperature, beam_size, end_symbol, device, sequence), beam):
                if finished:
                    if size > 100:
                        #if item not in [beam[0] for beam in finished_beams]:
                        finished_beams.append((item, score, size))
                else:
                    beam_candidates.extend(item)
                    #beam_candidates.append((item, score, size, sequence))
        
        beam_candidates.sort(key=lambda x: x[1])
        beam = beam_candidates[:beam_size]
        
        if len(beam) == 0:
            break
        else:
            print("Index:" + str(i))

    if not finished_beams:
        finished_beams = beam

    valid_sequences = select_possible_plans(sdr_idea_deserializer, occupiedNodes, initial_node, closest_nodes_from_goal_tag, action, finished_beams)    

    if not valid_sequences:
        finished_beams.sort(key=lambda x: x[1])
        ys, _, _ = finished_beams[-1]
    else:
        valid_sequences.sort(key=lambda x: x[1])
        ys, _, _ = valid_sequences[-1]

    return ys

def select_possible_plans(sdr_idea_deserializer, occupiedNodes, initial_node, closest_nodes_from_goal_tag, action, finished_beams):
    print("Amount of Plans Available:", len(finished_beams))

    invalid_plan_count = 0
    valid_plan_count = 0
    unconverted_plan_count = 0

    valid_sequences = []
    index = 0
    for ys, score, size in finished_beams:
        try:
            idea_array = convert_to_idea_array(ys, sdr_idea_deserializer)

            if is_valid_sequence(idea_array):
                if is_valid_plan(action=action, initial_node=initial_node, plan_steps=idea_array, occupiedNodes=occupiedNodes, closest_nodes_from_goal_tag=closest_nodes_from_goal_tag):
                    if (ys, score, size) not in valid_sequences:  # Check for duplicates
                        valid_sequences.append((ys, score, size))

                        print("\nPlan Index:", index)

                        for i in range(len(idea_array)):
                            print(
                                f'{i} - {idea_array[i].name} - {idea_array[i].value}')

                        print("Score:", score)
                        print("Total of Steps:", len(idea_array))

                        valid_plan_count += 1
                else:
                    print("\nINVALID Plan Index:", index)

                    for i in range(len(idea_array)):
                        print(
                            f'{i} - {idea_array[i].name} - {idea_array[i].value}')
    
                    print("Score:", score)
                    print("Total of Steps:", len(idea_array))
                    invalid_plan_count+=1
            else:
                print("\nINVALID Plan Index:", index)

                for i in range(len(idea_array)):
                    print(
                        f'{i} - {idea_array[i].name} - {idea_array[i].value}')

                print("Score:", score)
                print("Total of Steps:", len(idea_array))

                invalid_plan_count+=1

        except Exception as e:
            unconverted_plan_count += 1
            pass

        index += 1

    print("Total of Plans Correct:" + str(valid_plan_count))    
    print("Total of Plans Incorrect:" + str(invalid_plan_count))    
    print("Total of Unconverted Plans:" + str(unconverted_plan_count))
    return valid_sequences
