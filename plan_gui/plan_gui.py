from tkinter import *
from tkinter import ttk
from tkinter.filedialog import askopenfile 
from matplotlib import pyplot as plt, patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
import torch
import numpy as np
import torch.nn.functional as F

from gan_model.generator import CustomTransformerWithResiduals, beam_search
from pyctm.correction_engines.naive_bayes_correction_engine import NaiveBayesCorrectorEngine
from pyctm.representation.array_dictionary import ArrayDictionary
from pyctm.representation.dictionary import Dictionary
from pyctm.representation.idea import Idea
from pyctm.representation.sdr_idea_array import SDRIdeaArray
from pyctm.representation.sdr_idea_array_deserializer import SDRIdeaArrayDeserializer
from pyctm.representation.sdr_idea_array_serializer import SDRIdeaArraySerializer
from pyctm.representation.sdr_idea_deserializer import SDRIdeaDeserializer
from pyctm.representation.sdr_idea_serializer import SDRIdeaSerializer

def open_and_draw_graph_window(gui=None):
    
    global new_window
    global ax
    global figure

    new_window = Toplevel(gui)
 
    new_window.title("Graph Map")
    new_window.geometry("800x800")

    figure = plt.Figure(figsize=(15, 15), dpi=100)
    ax = figure.add_subplot()
    ax.set_xlim([-15, 15])
    ax.set_ylim([-15, 15])

    for node in graph['nodes']:
        draw_arrow(ax, node, graph['nodes'])
    
    for node in graph['nodes']:
        draw_node(ax, node)        


def open_json_file(gui=None):
    global graph    

    file_path = askopenfile(mode='r', filetypes=[('Json File', '.json')])
    if file_path is not None:
        graph = json.load(file_path)
        clear_button['state'] = 'normal'

def open_artags_json_file(gui=None):
    global artags

    file_path = askopenfile(mode='r', filetypes=[('Json File', '.json')])
    if file_path is not None:
        artags=json.load(file_path)

def open_dictionary_json_file(gui=None):
    file_path = askopenfile(mode='r', filetypes=[('Json File', '.json')])
    if file_path is not None:
        object=json.load(file_path)
        dictionary = ArrayDictionary(**object)
        sdr_idea_serializer.dictionary = dictionary
        sdr_idea_deserializer.dictionary = dictionary

def open_model_file(gui=None):
    global generator_model

    file_path = askopenfile(mode='r', filetypes=[('Pytorch Model File', '.pth')])
    if file_path is not None:
        vocabulary_size = 34
        d_model = 512 
        nhead = 8  
        num_encoder_layers = 2 
        num_decoder_layers = 4 
        dim_feedforward = 512  
        dropout = 0.3  
        max_seq_len = 626

        generator_model = CustomTransformerWithResiduals(vocabulary_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, max_seq_len)
        generator_model.load_state_dict(torch.load(file_path.name, map_location=torch.device('cpu')))
        generator_model.eval()
        if clear_button['state'] == 'normal':
            check_button['state'] = 'normal'

def draw_arrow(ax, start_node, nodes):
    start_coordinates = start_node['coordinates']

    for connection in start_node['connected']:
        end_node = nodes[connection-1]
        end_coordinates = end_node['coordinates']
        arrow = patches.FancyArrow(start_coordinates[0], start_coordinates[1], end_coordinates[0]-start_coordinates[0], end_coordinates[1]-start_coordinates[1])
        ax.add_patch(arrow)

def draw_node(ax, node):
    coordinates = node['coordinates']
    circle = patches.Circle((coordinates[0], coordinates[1]), radius=1, color='gray')
    ax.add_patch(circle)
    ax.annotate(node['id'], xy=(coordinates[0], coordinates[1]), fontsize=12, ha="center")

def random_value(gui=None):
    action_var.set("PICK" if np.random.randint(0,2) == 1 else "PLACE")
    init_position_var.set(str(float(np.random.randint(1, 17) if action_var.get() == 'PICK' else np.random.randint(1, 188))))
    goal_tag_var.set(str(float(np.random.randint(0, 188))))
    relative_position_var.set(str(float(np.random.randint(1, 5))))

def clear_board(gui=None):

    if new_window is not None:
        new_window.destroy()
    
    open_and_draw_graph_window(gui)

    figure_canvas = FigureCanvasTkAgg(figure, new_window)
    figure_canvas.get_tk_widget().pack(side=LEFT, fill=BOTH)

def create_gui():
    gui = Tk()
    gui.title("Plan Generated Test.")
    gui.geometry("680x320")

    return gui

def create_text_field(gui, row, label):
    label_text = Label(gui, text=label, anchor='w')
    label_text.grid(row=row, column=0)

    current_var = StringVar()

    entry = Entry(gui, textvariable=current_var)
    entry.grid(row=row, column=1, sticky='W')

    return current_var

def create_text_field_with_column(gui, row, column, label):
    label_text = Label(gui, text=label, anchor='w')
    label_text.grid(row=row, column=column)

    current_var = StringVar()

    entry = Entry(gui, textvariable=current_var)
    entry.grid(row=row, column=column+1, sticky='W')

    return current_var

def create_combo_box(gui, row, column, label, list):

    label_text = Label(gui, text=label, anchor='w')
    label_text.grid(row=row, column=column)

    current_var = StringVar()

    combobox = ttk.Combobox(gui, textvariable=current_var)
    combobox["values"] = list

    combobox.grid(row=row, column=column+1, sticky='W')

    return current_var

def create_check_box(gui, row, label):

    label_text = Label(gui, text=label, anchor='w')
    label_text.grid(row=row, column=0)

    current_var = StringVar()

    checkbox = ttk.Checkbutton(gui, textvariable=current_var, onvalue=True, offvalue=False)
    checkbox.grid(row=row, column=1, sticky='W')

    return current_var

def create_multiples_check_box(gui, row):
    for i in range(16):
        ttk.Checkbutton(gui, text=str(i+1), variable=check_vars[i], command=update_checked_numbers).grid(row=row+i//4, column=i%4, sticky='w')

def create_label(gui, row, label):
    label_text = Label(gui, text=label, anchor='w')
    label_text.grid(row=row, column=0)

def create_button(gui, row, column, text, func=None, state=None):
    button = Button(gui, text=text, command=lambda:func(gui) if func else None, state=state if state else 'normal')
    button.grid(row=row, column=column, pady=3)

    return button

def update_checked_numbers():
    global checked_numbers_var

    checked_numbers_var.clear()
    
    for i, var in enumerate(check_vars):
        if var.get():
            checked_numbers_var.append(float(i+1))

def check_state_plan(gui=None):

    if new_window is not None:
        new_window.destroy()
    
    open_and_draw_graph_window(gui)

    goal_idea = create_goal_idea()

    sdr_goal_idea = sdr_idea_serializer.serialize(goal_idea)

    converted_sdr = []
    for element in sdr_goal_idea.sdr:
        converted_sdr.append(float(element))
    
    sdr_goal_tensor = torch.from_numpy(np.array(converted_sdr)).view(1, 626)
    sdr_goal_tensor = sdr_goal_tensor.long()

    generate_plan_tensor = beam_search(model=generator_model, src=sdr_goal_tensor, start_symbol=1, end_symbol=2, max_len=626, beam_size=1, temperature=1, device='cpu')
    generate_plan = generate_plan_tensor[0].cpu().numpy().tolist()

    plan_idea_array = SDRIdeaArray(10, 7, 0)
    plan_idea_array.sdr = generate_plan

    plan_idea = sdr_idea_deserializer.deserialize(plan_idea_array)

    list_actions = [plan_idea]
    print(f'{plan_idea.name} - {plan_idea.value}')

    for i in range(len(plan_idea.child_ideas)):
        print(f'{plan_idea.child_ideas[i].name} - {plan_idea.child_ideas[i].value}')
        list_actions.append(plan_idea.child_ideas[i])

    print("Total of Steps:" + str(len(list_actions)))

    draw_plan(list_actions=list_actions)

    figure_canvas = FigureCanvasTkAgg(figure, new_window)
    figure_canvas.get_tk_widget().pack(side=LEFT, fill=BOTH)

def draw_plan(list_actions):
    
    previous_idea = None

    for i in range(len(list_actions)):
        if list_actions[i].name != 'stop' and list_actions[i].name != 'idle':
            
            idea_pose = get_position_from_idea(list_actions[i])
            
            if i != 0 and list_actions[i-1].name != 'idle':
                previous_idea = list_actions[i-1]
            else:
                if action_var.get() == 'PICK':
                    nodes = graph["nodes"]
                    for node in nodes:
                        if float(node["id"]) == float(init_position_var.get()):
                           draw_line((node['coordinates'])[0], (node['coordinates'])[1], idea_pose[0], idea_pose[1], 'r')
                           draw_point((node['coordinates'])[0], (node['coordinates'])[1], "%i" % int(float(init_position_var.get())), 'green', True)
                else:
                    artag = get_artag(float(init_position_var.get()))
                    draw_line((artag['pose'])[0], (artag['pose'])[1], idea_pose[0], idea_pose[1], 'r')
                    draw_point((artag['pose'])[0], (artag['pose'])[1], "%i" % int(float(init_position_var.get())), 'green', True)

            if previous_idea is not None:
                previous_idea_pose = get_position_from_idea(previous_idea)

                draw_line(previous_idea_pose[0], previous_idea_pose[1], idea_pose[0], idea_pose[1], 'r')

            if 'moveToNode' in list_actions[i].name:    
                draw_point(idea_pose[0], idea_pose[1], "%i" % int(list_actions[i].value), 'gold', True)
            
            elif 'moveTo' in list_actions[i].name:
                draw_point(idea_pose[0], idea_pose[1], "%i" % int(list_actions[i].value), 'orange', True)
            
            elif 'pick' in list_actions[i].name:
                draw_point(idea_pose[0], idea_pose[1], "%i" % int(list_actions[i].value[0]), 'purple', True)
            
            elif 'place' in list_actions[i].name:
                draw_point(idea_pose[0], idea_pose[1], "%i" % int(list_actions[i].value[0]), 'red', True)
        
def get_position_from_idea(idea):
    if idea is not None:
        if 'moveToNode' in idea.name:
            for node in graph['nodes']:
                if node['id'] == int(idea.value):
                    return node['coordinates']

        elif 'moveTo' in idea.name:
            return get_artag(idea.value)['pose']

        elif 'pick' in idea.name:
            return get_artag(idea.value[0])['pose']
            
        elif 'place' in idea.name:
            return get_artag(idea.value[0])['pose']
    
    return None

def get_artag(artag_id):
    
    for artag in artags:
        if artag["id"] == int(round(artag_id)):
            return artag
    
    return None

def draw_point(x, y, text, color, fill, radius=0.66):
    circle = patches.Circle((x, y), radius=radius, color=color, fill=fill)
    ax.add_patch(circle)
    ax.annotate(text, xy=(x, y), fontsize=12, ha="center", color='black', weight="bold")

def draw_line(x_i, y_i, x_f, y_f, color):
    arrow = patches.FancyArrow(x_i, y_i, x_f-x_i, y_f-y_i, color = color)
    ax.add_patch(arrow)

def create_goal_idea():
    goal_idea = Idea(_id=0, name="goal", value="", _type=1)

    init_node_idea = None
    goal_action_idea = None

    if action_var.get() == 'PICK':
        init_node_idea = Idea(_id=1, name="initialNode", value=float(init_position_var.get()), _type=1)
        goal_action_idea = Idea(_id=2, name="goalAction", value=2.0, _type=1)
    else:
        init_node_idea = Idea(_id=1, name="initialTag", value=float(init_position_var.get()), _type=1)
        goal_action_idea = Idea(_id=2, name="goalAction", value=3.0, _type=1)

    
    goal_tag_idea = Idea(_id=3, name="goalTag", value=float(goal_tag_var.get()), _type=1)
    goal_slot_idea = Idea(_id=4, name="goalSlot", value=float(relative_position_var.get()), _type=1)
    total_steps_idea = Idea(_id=5, name="totalSteps", value=float(plan_total_steps_var.get()), _type=1)

    occupied_nodes_idea = None
    if len(checked_numbers_var) == 0:
        occupied_nodes_idea = Idea(_id=6, name="occupiedNodes", value="", _type=1)    
    else:
        occupied_nodes_idea = Idea(_id=6, name="occupiedNodes", value=[float(i) for i in checked_numbers_var], _type=1)
    
    goal_idea.add(init_node_idea)
    goal_idea.add(goal_action_idea)
    goal_idea.add(goal_tag_idea)
    goal_idea.add(goal_slot_idea)
    goal_idea.add(total_steps_idea)
    goal_idea.add(occupied_nodes_idea)

    return goal_idea

def compare_sdr(goal, target):
        for i in range(10):
            print("Channel:" + str(i) + " OK!")
            for j in range(32):
                for k in range(32):
                    if goal[i,j,k] != target[0][i][j][k]:
                        print("Channel:" + str(i))
                        print("Row:" + str(j))
                        print("Collumn:" + str(k))
                        return False
                    
        return True

def goal_intention_value(goal_intention):
    if goal_intention == 'TRANSPORT':
        return float(1.0)
    elif goal_intention == 'CHARGE':
        return float(2.0)
    else:
        return float(3.0)

if __name__ == '__main__':
    gui = create_gui()

    global new_window
    global ax
    global figure

    global action_var
    global init_position_var
    global goal_tag_var
    global relative_position_var
    global plan_total_steps_var

    global checked_numbers_var
    checked_numbers_var = []

    global check_vars 
    check_vars = [BooleanVar() for _ in range(16)]


    global clear_button
    global check_button
    global random_button

    global sdr_idea_serializer
    global sdr_idea_deserializer
    global artags

    new_window = None  
    ax = None
    figure = None

    sdr_idea_serializer = SDRIdeaArraySerializer(total_of_ideas=10, total_of_values=7, default_value=0)
    sdr_idea_deserializer = SDRIdeaArrayDeserializer(sdr_idea_serializer.dictionary)

    create_label(gui, 0, "Occupied Nodes:")
    create_multiples_check_box(gui, 1)

    action_var = create_combo_box(gui, 7, 0, "Action:", ['PICK', 'PLACE'])
    action_var.set("PICK")

    init_position_var = create_text_field_with_column(gui, 8, 0, "Initial Tag/Node:")
    init_position_var.set("1.0")

    goal_tag_var = create_text_field_with_column(gui, 8, 2, "Goal Tag:")
    goal_tag_var.set("71.0")

    relative_position_var = create_text_field_with_column(gui, 9, 0, "Goal Relative Position:")
    relative_position_var.set("1.0")

    plan_total_steps_var = create_text_field_with_column(gui, 9, 2, "Plan Total Steps:")
    plan_total_steps_var.set("10.0")
    
    create_button(gui, 10, 0, "Load Graph File", open_json_file)
    create_button(gui, 10, 1, "Load ARTags File", open_artags_json_file)
    create_button(gui, 10, 2, "Load Dictionary File", open_dictionary_json_file)
    create_button(gui, 10, 3, "Load Planner Model File", open_model_file)
    
    clear_button = create_button(gui, 11, 1, "Clear Board", clear_board, 'disabled')
    check_button = create_button(gui, 11, 2, "Check Plan", check_state_plan, 'disabled')

    random_button = create_button(gui, 11, 3, "Random", random_value, 'normal')

    gui.mainloop()

    