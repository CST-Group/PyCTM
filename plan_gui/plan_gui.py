from tkinter import *
from tkinter import ttk
from tkinter.filedialog import askopenfile 
from matplotlib import pyplot as plt, patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
import torch
import numpy as np
import torch.nn.functional as F

from gan_model.generator import Generator
from pyctm.correction_engines.naive_bayes_correction_engine import NaiveBayesCorrectorEngine
from pyctm.representation.dictionary import Dictionary
from pyctm.representation.idea import Idea
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
        dictionary = Dictionary(**object)
        sdr_idea_serializer.dictionary = dictionary
        sdr_idea_deserializer.dictionary = dictionary

def open_model_file(gui=None):
    global generator_model

    file_path = askopenfile(mode='r', filetypes=[('Pytorch Model File', '.pth')])
    if file_path is not None:
        generator_model = Generator(in_channels=10, features=128, image_size=32, control_size=32)
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


def clear_board(gui=None):

    if new_window is not None:
        new_window.destroy()
    
    open_and_draw_graph_window(gui)

    figure_canvas = FigureCanvasTkAgg(figure, new_window)
    figure_canvas.get_tk_widget().pack(side=LEFT, fill=BOTH)

def create_gui():
    gui = Tk()
    gui.title("Plan Generated Test.")
    gui.geometry("400x380")

    return gui

def create_text_field(gui, row, label):
    label_text = Label(gui, text=label, anchor='w')
    label_text.grid(row=row, column=0)

    current_var = StringVar()

    entry = Entry(gui, textvariable=current_var)
    entry.grid(row=row, column=1, sticky='W')

    return current_var

def create_combo_box(gui, row, label, list):

    label_text = Label(gui, text=label, anchor='w')
    label_text.grid(row=row, column=0)

    current_var = StringVar()

    combobox = ttk.Combobox(gui, textvariable=current_var)
    combobox["values"] = list

    combobox.grid(row=row, column=1, sticky='W')

    return current_var

def check_state_plan(gui=None):

    prepare_correction_engine()
    list_actions = []

    if new_window is not None:
        new_window.destroy()
    
    open_and_draw_graph_window(gui)

    goal_idea = create_goal_idea()

    sdr_goal_idea = sdr_idea_serializer.serialize(goal_idea)

    sdr_goal_tensor = torch.from_numpy(sdr_goal_idea.sdr).view(1, 10, 32, 32)
    sdr_goal_tensor = sdr_goal_tensor.float()

    control =  torch.from_numpy(np.asarray([float(plan_size_var.get())] + [float(i) for i in occupied_nodes_var.get().split(',')])).float()
    control = control.unsqueeze(0)

    control = F.pad(control, (0, 32 - control.size(1)), "constant", 0.0).float()

    sdr_action_step_tensor = torch.argmax(generator_model(sdr_goal_tensor, control).view(1, 2, 32, 32).detach(), dim=1).view(1, 1, 32, 32)

    action_step_idea = sdr_idea_deserializer.deserialize(sdr_action_step_tensor[0].detach().numpy())
    
    id_index = 6

    action_step_idea.id = id_index
    action_step_idea.type = 1.0

    goal_idea.child_ideas[-1].add(action_step_idea)
    list_actions.append(action_step_idea)

    print("%s - Action: %s - Value: %s" % (action_step_idea.id, action_step_idea.name, action_step_idea.value))

    for i in range(20):
        if action_step_idea.name != 'stop':

            sdr_goal_idea = sdr_idea_serializer.serialize(goal_idea)

            sdr_goal_tensor = torch.from_numpy(sdr_goal_idea.sdr).view(1, 10, 32, 32)   
            sdr_goal_tensor = sdr_goal_tensor.float()

            # full_control = torch.randn(1, 64).float()
            sdr_action_step_tensor = torch.argmax(generator_model(sdr_goal_tensor, control).view(1, 2, 32, 32).detach(), dim=1).view(1, 1, 32, 32)

            new_action_step_idea = sdr_idea_deserializer.deserialize(sdr_action_step_tensor[0].detach().numpy())
            id_index+=1

            new_action_step_idea.id = id_index
            new_action_step_idea.type = 1.0

            action_step_idea = new_action_step_idea

            goal_idea.child_ideas[-1].add(action_step_idea)

            if len(goal_idea.child_ideas[-1].child_ideas) > 5:
                goal_idea.child_ideas[-1].child_ideas.pop(0)
            
            list_actions.append(action_step_idea)

            print("%s - Action: %s - Value: %s" % (new_action_step_idea.id, new_action_step_idea.name, new_action_step_idea.value))
        
        else:
            break

    
    print("Total of Steps:" + str(len(list_actions) + 1))

    draw_plan(list_actions=list_actions)

    figure_canvas = FigureCanvasTkAgg(figure, new_window)
    figure_canvas.get_tk_widget().pack(side=LEFT, fill=BOTH)

def prepare_correction_engine():
    correction_engine = NaiveBayesCorrectorEngine(sdr_idea_serializer.dictionary)
    sdr_idea_deserializer.corrector_engine = correction_engine


def save_to_test(sdr_plan_tensor):
    plan_generated_dic = {
        'realPlan': sdr_plan_tensor.view(16,32,32).detach().tolist(),
        'fakePlan': []
    }

    with open('./pix2pix_plan_generated_local.json', 'w') as write_file:
        json.dump(plan_generated_dic, write_file)


def draw_plan(list_actions):
    
    previous_idea = None

    for i in range(len(list_actions)):
        if list_actions[i].name != 'stop' and list_actions[i].name != 'idle':
            
            idea_pose = get_position_from_idea(list_actions[i])
            
            if i != 0 and list_actions[i-1].name != 'idle':
                previous_idea = list_actions[i-1]
            else:
                nodes = graph["nodes"]
                for node in nodes:
                    if float(node["id"]) == float(init_node_var.get()):
                       draw_line((node['coordinates'])[0], (node['coordinates'])[1], idea_pose[0], idea_pose[1], 'r')
                       draw_point((node['coordinates'])[0], (node['coordinates'])[1], "%i" % int(float(init_node_var.get())), 'green', True)

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
    #goal_idea = Idea(_id=0, name="Goal", value=float(goal_intention_value(goal_intention_var.get())), _type=0.0)
    goal_idea = Idea(_id=0, name="Goal", value="", _type=0)
    init_node_idea = Idea(_id=1, name="initialNode", value=float(init_node_var.get()), _type=1)
    intermidiate_idea = Idea(_id=2, name="middleTarget", value=[float(i) for i in intermidiate_var.get().split(',')], _type=1)
    final_goal_idea = Idea(_id=3, name="finalTarget", value=[float(i) for i in final_goal_var.get().split(',')], _type=1)
    context_idea = Idea(_id=4, name="actionSteps", value="", _type=0)
    
    goal_idea.add(init_node_idea)
    goal_idea.add(intermidiate_idea)
    goal_idea.add(final_goal_idea)

    context_idea.add(Idea(_id=5, name='idle', value="", _type=1))

    goal_idea.add(context_idea)

    return goal_idea


def get_last_action_step_sdr(last_action_step_idea):

    sdr_idea_serializer_local = SDRIdeaSerializer(1, 32, 32)
    sdr_idea_serializer_local.dictionary = sdr_idea_serializer.dictionary

    return sdr_idea_serializer_local.serialize(last_action_step_idea)
    

def create_button(gui, row, column, text, func=None, state=None):
    button = Button(gui, text=text, command=lambda:func(gui) if func else None, state=state if state else 'normal')
    button.grid(row=row, column=column, pady=3)

    return button

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

    global init_pose_var
    global middle_pose_var
    global goal_pose_var
    global goal_intention_var
    global plan_size_var
    global occupied_nodes_var


    global clear_button
    global check_button

    global sdr_idea_serializer
    global sdr_idea_deserializer
    global artags

    new_window = None  
    ax = None
    figure = None

    sdr_idea_serializer = SDRIdeaSerializer(10, 32, 32)
    sdr_idea_deserializer = SDRIdeaDeserializer(sdr_idea_serializer.dictionary)
    
    init_node_var = create_text_field(gui, 0, "Initial Node:")
    init_node_var.set("1.0")

    occupied_nodes_var = create_text_field(gui, 1, "Occupied Nodes:")    
    occupied_nodes_var.set("0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0")

    plan_size_var = create_text_field(gui, 2, "Plan Size:")
    plan_size_var.set("10.0")

    intermidiate_var = create_text_field(gui, 3, "Middle Target:")
    intermidiate_var.set("2.0,71.0,1.0")

    final_goal_var = create_text_field(gui, 4, "Final Target:")   
    final_goal_var.set("3.0,62.0,1.0")

    # goal_intention_var = create_combo_box(gui, 3, "Goal Intention:", ['EXPLORATION', 'CHARGE', 'TRANSPORT'])
    
    
    create_button(gui, 5, 0, "Load Graph File", open_json_file)
    create_button(gui, 5, 1, "Load ARTags File", open_artags_json_file)
    create_button(gui, 6, 0, "Load Dictionary File", open_dictionary_json_file)
    create_button(gui, 6, 1, "Load Planner Model File", open_model_file)
    
    clear_button = create_button(gui, 7, 0, "Clear Board", clear_board, 'disabled')
    check_button = create_button(gui, 7, 1, "Check Plan", check_state_plan, 'disabled')

    gui.mainloop()

    