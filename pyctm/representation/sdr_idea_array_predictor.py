
from concurrent.futures import ThreadPoolExecutor
import math
import torch
import torch.nn.functional as F
from pyctm.representation.sdr_idea_array import SDRIdeaArray


class SDRIdeaArrayPredictor:

    def __init__(self, sdr_idea_deserializer):
        self.sdr_idea_deserializer = sdr_idea_deserializer

    def beam_search(self, model, src, start_symbol, end_symbol, max_len, beam_size, temperature, sdr_idea_deserializer, device, occupiedNodes, initial_node, closest_nodes_from_goal_tag, action):
        src = src.to(device)
        ys = torch.LongTensor([[start_symbol]]).type_as(src.data)
        states = self.get_states(False)
        steps = self.get_steps(states[0])
        beam = [(ys, 0.0, 7.0, False, states, steps)]
        finished_beams = []
    
        for i in range(max_len):
        
            beam_candidates = []
            with ThreadPoolExecutor(max_workers=beam_size) as executor:
                for answer, score, finished, size, states, steps in executor.map(
                    lambda answer: self.process_beam_item(answer, src, model, temperature, beam_size, end_symbol, device), beam):
                    if finished:
                        if size > 100:
                            finished_beams.append((answer, score, finished, size, states, steps))
                    else:
                        beam_candidates.extend(answer)
            
            beam_candidates.sort(key=lambda x: x[1])
            beam = beam_candidates[:beam_size]
            
            if len(beam) == 0:
                break
            else:
                print("Index:" + str(i))
    
        if not finished_beams:
            finished_beams = beam
    
        valid_sequences = self.select_possible_plans(sdr_idea_deserializer, occupiedNodes, initial_node, closest_nodes_from_goal_tag, action, finished_beams)    
    
        if not valid_sequences:
            finished_beams.sort(key=lambda x: x[1])
            ys, _, _ = finished_beams[-1]
        else:
            valid_sequences.sort(key=lambda x: x[1])
            ys, _, _ = valid_sequences[-1]
    
        return ys
    
    def process_beam_item(self, item, src, model, temperature, beam_size, end_symbol, device='cpu'):
        answer, score, _, _, states, steps = item
        if answer[-1, -1].eq(end_symbol).item():
            return answer, score, True, answer.shape[1], states, steps
    
        current_state, current_step, states, steps, allowed_tokens = self.get_allowed_tokens(states, steps) 
    
        with torch.no_grad():
            logits = model(src, answer)[-1, :]
        logits.div_(temperature) 
        probs = F.softmax(logits, dim=-1)
    
        if allowed_tokens is not None:
            for index in range(len(probs[0,:])):
                if index not in allowed_tokens:
                    probs[:, index] = 0.0
    
        top_limit = min(beam_size, len(allowed_tokens))
        top_probs, top_ix = probs.topk(top_limit)
    
        candidates = []
        for i in range(top_limit):
            prob = top_probs[-1, i].item()
            ix = top_ix[-1, i].item()
            next_answer = torch.cat([answer, torch.tensor([[ix]], device=device)], dim=1)
            next_score = score - math.log(prob)

            new_steps = steps.copy()
            new_states = states.copy()

            self.validate_and_extend_steps(new_states, new_steps, current_state, next_answer)

            candidates.append((next_answer, next_score, False, next_answer.shape[1], new_states, new_steps))

        return candidates, score, False, next_answer.shape[1], states, steps

    def validate_and_extend_steps(self, states, steps, current_state, next_answer):
        if len(steps) == 0:
            if current_state == "LENGTH":
                length_value = self.sdr_idea_deserializer.get_local_numeric_value(next_answer[0, -6:].tolist())
                    
                if length_value <= 0:
                    length_value = 1
                    
                metadata_type = self.sdr_idea_deserializer.get_metadata_type(int(self.sdr_idea_deserializer.get_local_string_value(next_answer[:, -7].item())))
                if metadata_type == "STRING_ARRAY" or metadata_type == "STRING_VALUE":
                    steps.extend(int(length_value) * self.get_steps("STRING_VALUE"))
                elif metadata_type == "NUM_ARRAY" or metadata_type == "NUM_VALUE":
                    steps.extend(int(length_value) * self.get_steps("NUM_VALUE"))

                states.pop(0)
                states.append("VALUE")

    def get_allowed_tokens(self, states, steps):
        current_state = None
        current_step = None

        if len(steps) == 0:
            states.pop(0) 

            if len(states) == 0:
                states = self.get_states(True)
               
            current_state = states[0]
            steps = self.get_steps(current_state)
            
        if current_state is None:
            current_state = states[0]

        current_step = steps.pop(0)
        allowed_tokens = self.get_step_index()[current_step]
            
        return current_state, current_step, states, steps, allowed_tokens


    def get_steps(self, state):
        
        steps = {
            "PARENT_ID": ["END", "NUMBER", "NUMBER", "SIGNAL", "NUMBER", "SIGNAL"],
            "ID": ["NUMBER", "NUMBER", "NUMBER", "SIGNAL", "NUMBER", "SIGNAL"],
            "NAME": ["STRING"],
            "TYPE": ["TYPE"],
            "METADATA": ["METADATA"],
            "LENGTH": ["NUMBER", "NUMBER", "NUMBER", "SIGNAL", "NUMBER", "SIGNAL"],
            "NUM_VALUE": ["NUMBER", "NUMBER", "NUMBER", "SIGNAL", "NUMBER", "SIGNAL"],
            "STRING_VALUE": ["STRING"],
            "END": ["END"]
        }
    
        return steps[state]


    def get_states(self, is_parent=False):
        sequence = ["ID", "NAME", "TYPE", "METADATA", "LENGTH"]
        if is_parent:
            sequence.insert(0, "PARENT_ID")

        return sequence

    def get_step_index(self):
        states = {
            "NUMBER": [6,7,8,9,10,11,12,13,14,15],
            "SIGNAL": [4,5],
            "STRING": [16,19,20,22,23,24,25,27,28,29,30,31,32],
            "TYPE": [17],
            "METADATA": [17, 18, 21, 26],
            "SPECIAL": [2],
            "END": [2,6,7,8,9,10,11,12,13,14,15]
        }

        return states;


    def select_possible_plans(self, sdr_idea_deserializer, occupiedNodes, initial_node, closest_nodes_from_goal_tag, action, finished_beams):
        print("Amount of Plans Available:", len(finished_beams))

        invalid_plan_count = 0
        valid_plan_count = 0
        unconverted_plan_count = 0

        valid_sequences = []
        index = 0
        for ys, score, size, _, _, _ in finished_beams:
            try:
                idea_array = self.convert_to_idea_array(ys, sdr_idea_deserializer)

                if self.is_valid_sequence(idea_array):
                    if self.is_valid_plan(action=action, initial_node=initial_node, plan_steps=idea_array, occupiedNodes=occupiedNodes, closest_nodes_from_goal_tag=closest_nodes_from_goal_tag):
                        #if (ys, score, size) not in valid_sequences:  # Check for duplicates
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

    def get_graph_connection(self):
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

    def is_valid_plan(self, action, initial_node, closest_nodes_from_goal_tag, plan_steps, occupiedNodes=[]):
        if action == 'PICK':
            current_step = plan_steps[0]
            if current_step.name == "moveToNode":
                if float(current_step.value) != float(initial_node):
                    current_node = str(current_step.value)
                    initial_node = str(initial_node)
                    if current_node not in self.get_graph_connection()[initial_node]:
                        return False

        for i in range(len(plan_steps) - 1):
            current_step = plan_steps[i]
            next_step = plan_steps[i + 1]

            if current_step.name == "moveToNode" and next_step.name == "moveToNode":
                try:
                    current_node = str(current_step.value)
                    next_node = str(next_step.value)
                    if next_node not in self.get_graph_connection()[current_node]:
                        return False

                    if current_step.value in occupiedNodes:
                        return False
                except Exception as e:
                    return False

        return True  

    def convert_to_idea_array(self, tensor, sdr_idea_deserializer):
        sdr_idea = tensor.squeeze(0).detach().cpu().numpy().tolist()

        plan_idea_array = SDRIdeaArray(10, 7, 0)
        plan_idea_array.sdr = sdr_idea

        action_step_idea = sdr_idea_deserializer.deserialize(plan_idea_array)

        full_goal = [action_step_idea]
        for i in range(len(action_step_idea.child_ideas)):
            full_goal.append(action_step_idea.child_ideas[i])

        return full_goal

    def is_valid_sequence(self, idea_array):
        for idea in idea_array:
            if not self.is_valid_idea(idea):
                return False
        return True
    
    def is_valid_idea(self, step_idea):
        if step_idea.name == "pick" or step_idea.name == "place":
            if (isinstance(step_idea.value, list) and
                len(step_idea.value) == 2 and
                0 <= step_idea.value[0] <= 187 and
                1 <= step_idea.value[1] <= 4):
                return True
            else:
                return False

        elif step_idea.name == "moveTo":
            if isinstance(step_idea.value, float) and 0 <= step_idea.value <= 187:
                return True
            else:
                return False

        elif step_idea.name == "moveToNode":
            if isinstance(step_idea.value, float) and 1 <= step_idea.value <= 16:
                return True
            else:
                return False


        return True