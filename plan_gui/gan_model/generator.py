import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer
import copy

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

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension (d_model) is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # Initialize dimensions
        self.d_model = d_model # Model's dimension
        self.num_heads = num_heads # Number of attention heads
        self.d_k = d_model // num_heads # Dimension of each head's key, query, and value

        self.W_q = nn.Linear(d_model, d_model) # Query transformation
        self.W_k = nn.Linear(d_model, d_model) # Key transformation
        self.W_v = nn.Linear(d_model, d_model) # Value transformation
        self.W_o = nn.Linear(d_model, d_model) # Output transformation

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = torch.softmax(attn_scores, dim=-1)

        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        output = self.W_o(self.combine_heads(attn_output))
        return output

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
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
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
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
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

class CustomTransformerWithResiduals(nn.Module):
    def __init__(self, vocabulary_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, max_seq_len=20):
        super(CustomTransformerWithResiduals, self).__init__()

        self.vocabulary_size = vocabulary_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Embedding layer with residual connections
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
        positional_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0)
        return nn.Parameter(positional_encoding, requires_grad=False)

    def forward(self, src, tgt):
        batch_size_tgt, seq_len_tgt = tgt.size()
        # batch_size, seq_len = src.size()

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
    

def beam_search(model, src, start_symbol, end_symbol, max_len, beam_size, temperature, device):
    src = src.to(device)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data).to(device)
    beam = [(ys, 0.0)]
    finished_beams = []

    for _ in range(max_len - 1):
        candidates = []
        for ys, score in beam:
            if ys[-1, -1].eq(end_symbol).item():
                finished_beams.append((ys, score))
                continue

            with torch.no_grad():
                logits = model(src, ys)[-1, :]
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            top_probs, top_ix = probs.topk(beam_size)

            for i in range(beam_size):
                prob = top_probs[-1, i].item()
                ix = top_ix[-1, i].item()
                next_ys = torch.cat([ys, torch.tensor([[ix]], device=device)], dim=1)
                next_score = score - math.log(prob)
                candidates.append((next_ys, next_score))

        beam = sorted(candidates, key=lambda x: x[1])[:beam_size]

    if not finished_beams:
        finished_beams = beam

    ys, _ = sorted(finished_beams, key=lambda x: x[1])[-1]
    return ys