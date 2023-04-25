from tkinter import *
from tkinter import ttk
from tkinter.filedialog import askopenfile 
from matplotlib import pyplot as plt, patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
import torch

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
        # generator_model = Generator(in_channels=5, features=64, image_size=32)

        # generator_model = Generator(in_channels=6, features=32, image_size=32)
        generator_model = Generator(in_channels=24, features=32, image_size=32)
        # generator_model = Generator(embed_size=1024, hidden_size=4096, num_layers=4, encoded_image_size=32, num_heads=16, dropout=0.1)
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

    if new_window is not None:
        new_window.destroy()
    
    open_and_draw_graph_window(gui)

    last_action_idea = Idea(_id=4, name='idle', value="", _type=0)

    goal_idea = create_idea([last_action_idea])

    sdr_goal_idea = sdr_idea_serializer.serialize(goal_idea)

    sdr_goal_tensor = torch.from_numpy(sdr_goal_idea.sdr).view(1, 24, 32, 32)
    sdr_goal_tensor = sdr_goal_tensor.float()

    # sdr_action_step_tensor = generator_model(sdr_goal_tensor).view(1, 1, 32, 32)
    sdr_action_step_tensor = torch.max(generator_model(sdr_goal_tensor).view(1, 2, 32, 32).detach(), dim=1).indices.view(1, 1, 32, 32)

    # sdr_action_step_tensor[sdr_action_step_tensor<0.5] = 0
    # sdr_action_step_tensor[sdr_action_step_tensor>=0.5] = 1

    action_step_idea = sdr_idea_deserializer.deserialize(sdr_action_step_tensor[0].detach().numpy())

    draw_idea(action_step_idea, None)

    id_index = 5

    for i in range(10):
        if action_step_idea.name != 'stop':
            action_step_idea.id = id_index
            goal_idea.child_ideas[-1].add(action_step_idea)

            sdr_goal_idea = sdr_idea_serializer.serialize(goal_idea)

            sdr_goal_tensor = torch.from_numpy(sdr_goal_idea.sdr).view(1, 24, 32, 32)
            sdr_goal_tensor = sdr_goal_tensor.float()

            sdr_action_step_tensor = torch.max(generator_model(sdr_goal_tensor).view(1, 2, 32, 32).detach(), dim=1).indices.view(1, 1, 32, 32)

            # sdr_action_step_tensor[sdr_action_step_tensor<0.5] = 0
            # sdr_action_step_tensor[sdr_action_step_tensor>=0.5] = 1

            new_action_step_idea = sdr_idea_deserializer.deserialize(sdr_action_step_tensor[0].detach().numpy())
            id_index = id_index + 1

            new_action_step_idea.id = id_index

            draw_idea(new_action_step_idea, action_step_idea)
            action_step_idea = new_action_step_idea
            
        
        else:
            break


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


def draw_idea(idea, previous_node):
    if idea is not None:
        print("%s - Action: %s - Value: %s" % (idea.id-5, idea.name, idea.value))

        if 'move' in idea.name:
            if previous_node is not None:
                draw_line(previous_node.value[0], previous_node.value[1], idea.value[0], idea.value[1], 'r')

            draw_point(idea.value[0], idea.value[1], idea.id, 'gold', True)
        elif 'pick' in idea.name:
            if previous_node is not None:
                draw_line(previous_node.value[0], previous_node.value[1], idea.value[0], idea.value[1], 'r')

            draw_point(idea.value[0], idea.value[1], 'PK', 'orange', True)
        elif 'place' in idea.name:
            if previous_node is not None:
                draw_line(previous_node.value[0], previous_node.value[1], idea.value[0], idea.value[1], 'r')

            draw_point(idea.value[0], idea.value[1], 'PC', 'purple', True)


def draw_point(x, y, text, color, fill, radius=0.66):
    circle = patches.Circle((x, y), radius=radius, color=color, fill=fill)
    ax.add_patch(circle)
    ax.annotate(text, xy=(x, y), fontsize=12, ha="center", color='black', weight="bold")

def draw_line(x_i, y_i, x_f, y_f, color):
    arrow = patches.FancyArrow(x_i, y_i, x_f-x_i, y_f-y_i, color = color)
    ax.add_patch(arrow)

def create_idea(list_action_steps_ideas):

    goal_idea = Idea(_id=0, name=goal_intention_var.get(), value="", _type=0)

    init_pose_idea = Idea(_id=1, name="robotPose", value=[float(i) for i in init_pose_var.get().split(',')], _type=1)
    middle_pose_idea = Idea(_id=2, name="pickPose", value=[float(i) for i in middle_pose_var.get().split(',')], _type=1)
    goal_pose_idea = Idea(_id=3, name="placePose", value=[float(i) for i in goal_pose_var.get().split(',')], _type=1)
    context_idea = Idea(_id=4, name="context", value="", _type=0)
    
    goal_idea.add(init_pose_idea)
    goal_idea.add(middle_pose_idea)
    goal_idea.add(goal_pose_idea)

    for action_step in list_action_steps_ideas:
        context_idea.add(action_step)

    goal_idea.add(context_idea)

    draw_point(init_pose_idea.value[0], init_pose_idea.value[1], 'I', 'green', True, radius=1)
    draw_point(goal_pose_idea.value[0], goal_pose_idea.value[1], 'F', 'red', True, radius=1)

    return goal_idea

def create_button(gui, row, column, text, func=None, state=None):
    button = Button(gui, text=text, command=lambda:func(gui) if func else None, state=state if state else 'normal')
    button.grid(row=row, column=column, pady=3)

    return button

if __name__ == '__main__':
    gui = create_gui()

    global new_window
    global ax
    global figure

    global init_pose_var
    global middle_pose_var
    global goal_pose_var
    global goal_intention_var

    global clear_button
    global check_button

    global sdr_idea_serializer
    global sdr_idea_deserializer

    new_window = None
    ax = None
    figure = None

    sdr_idea_serializer = SDRIdeaSerializer(24, 32, 32)
    sdr_idea_deserializer = SDRIdeaDeserializer(sdr_idea_serializer.dictionary)
    
    init_pose_var = create_text_field(gui, 0, "Init Pose:")
    middle_pose_var = create_text_field(gui, 1, "Middle Pose:")
    goal_pose_var = create_text_field(gui, 2, "Goal Pose:")    
    goal_intention_var = create_combo_box(gui, 3, "Goal Intention:", ['EXPLORATION', 'CHARGE', 'TRANSPORT'])
    
    create_button(gui, 4, 0, "Load Planner Model File", open_model_file)
    create_button(gui, 4, 1, "Load Graph File", open_json_file)

    create_button(gui, 5, 0, "Load Dictionary File", open_dictionary_json_file)

    clear_button = create_button(gui, 6, 0, "Clear Board", clear_board, 'disabled')
    check_button = create_button(gui, 6, 1, "Check Plan", check_state_plan, 'disabled')

    gui.mainloop()

    