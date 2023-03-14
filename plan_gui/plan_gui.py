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
        generator_model = Generator(in_channels=16, features=16, image_size=32)
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

    current_state_idea = create_idea()

    sdr_idea = sdr_idea_serializer.serialize(current_state_idea)

    sdr_tensor = torch.from_numpy(sdr_idea.sdr).view(1, 16, 32, 32)
    sdr_tensor = sdr_tensor.float()

    sdr_plan_tensor = generator_model(sdr_tensor)

    sdr_plan_tensor[sdr_plan_tensor<0.5] = 0
    sdr_plan_tensor[sdr_plan_tensor>=0.5] = 1

    # save_to_test(sdr_plan_tensor)

    plan_idea = sdr_idea_deserializer.deserialize(sdr_plan_tensor[0].detach().numpy())

    draw_idea(plan_idea.child_ideas[0], None)    

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
        print("%s - Action: %s - Value: %s" % (idea.id, idea.name, idea.value))

        if 'move' in idea.name:
            if previous_node is not None:
                draw_line(previous_node[0], previous_node[1], idea.value[0], idea.value[1], 'r')

            draw_point(idea.value[0], idea.value[1], idea.id, 'gold', True)
        elif 'pick' in idea.name:
            if previous_node is not None:
                draw_line(previous_node[0], previous_node[1], idea.value[0], idea.value[1], 'r')

            draw_point(idea.value[0], idea.value[1], 'PK', 'orange', True)
        elif 'place' in idea.name:
            if previous_node is not None:
                draw_line(previous_node[0], previous_node[1], idea.value[0], idea.value[1], 'r')

            draw_point(idea.value[0], idea.value[1], 'PC', 'purple', True)

        if idea.child_ideas is not None:
            if len(idea.child_ideas) > 0:
                if idea.value is not None:
                    if idea.value != '':
                        draw_idea(idea.child_ideas[0], [idea.value[0], idea.value[1]])

def draw_point(x, y, text, color, fill, radius=0.66):
    circle = patches.Circle((x, y), radius=radius, color=color, fill=fill)
    ax.add_patch(circle)
    ax.annotate(text, xy=(x, y), fontsize=12, ha="center", color='black', weight="bold")

def draw_line(x_i, y_i, x_f, y_f, color):
    arrow = patches.FancyArrow(x_i, y_i, x_f-x_i, y_f-y_i, color = color)
    ax.add_patch(arrow)

def create_idea():

    current_state_idea = Idea(_id=0, name='currentState', value="")

    activation_range_front_idea = Idea(_id=3, name='activationRangeFront', value=int(activ_front_var.get()), _type=1)
    activation_range_rear_idea = Idea(_id=4, name='activationRangeRear', value=int(activ_rear_var.get()), _type=1)
    is_battery_low_idea = Idea(_id=5, name='isBatteryLow', value=int(activ_rear_var.get()), _type=1)
    #robot_state_idea = Idea(_id=6, name='robotState', value = robot_state_var.get(), _type=1)
    #ur5_state_idea = Idea(_id=7, name='ur5State', value=ur5_state_var.get(), _type=1)
    mir_pose_idea = Idea(_id=6, name='mirPose', value=[float(i) for i in mir_pose_var.get().split(',')], _type=1)
    goal_intetion_idea = Idea(_id=7, name='goalIntention', value=goal_intention_var.get(), _type=1)
    goal_pose_idea = Idea(_id=8, name='goalPose', value=[float(i) for i in goal_pose_var.get().split(',')], _type=1)

    transport_requests_idea = None

    if transport_request_var.get() != '':
        transport_requests_idea = Idea(_id=1, name='transportRequests', value="", _type=0)
        transport_request_value = [str(i) for i in transport_request_var.get().split(',')]
        transport_requests_idea.add(Idea(_id=2, name=str(transport_request_value[0]), value=transport_request_value[1:], _type=1))

        current_state_idea.add(transport_requests_idea)

    current_state_idea.add(activation_range_front_idea)
    current_state_idea.add(activation_range_rear_idea)
    current_state_idea.add(is_battery_low_idea)
    #current_state_idea.add(robot_state_idea)
    #current_state_idea.add(ur5_state_idea)
    current_state_idea.add(mir_pose_idea)
    current_state_idea.add(goal_intetion_idea)
    current_state_idea.add(goal_pose_idea)
    
    
    
    

    draw_point(mir_pose_idea.value[0], mir_pose_idea.value[1], 'I', 'green', True, radius=1)
    draw_point(goal_pose_idea.value[0], goal_pose_idea.value[1], 'F', 'red', True, radius=1)

    return current_state_idea

def create_button(gui, row, column, text, func=None, state=None):
    button = Button(gui, text=text, command=lambda:func(gui) if func else None, state=state if state else 'normal')
    button.grid(row=row, column=column, pady=3)

    return button

if __name__ == '__main__':
    gui = create_gui()

    global new_window
    global ax
    global figure

    global mir_pose_var
    global ident_right_var
    global ident_left_var
    global activ_front_var
    global activ_rear_var
    global is_battery_var
    global robot_state_var
    global ur5_state_var

    global clear_button
    global check_button

    global sdr_idea_serializer
    global sdr_idea_deserializer

    new_window = None
    ax = None
    figure = None

    sdr_idea_serializer = SDRIdeaSerializer(16, 32, 32)
    sdr_idea_deserializer = SDRIdeaDeserializer(sdr_idea_serializer.dictionary)
    
    mir_pose_var = create_text_field(gui, 0, "MIR Pose:")
    ident_right_var = create_text_field(gui, 1, "Identified Markers Camera Right:")
    ident_left_var = create_text_field(gui, 2, "Identified Markers Camera Left:")
    activ_front_var = create_combo_box(gui, 3, "Activation Range Front:", [0, 1])
    activ_rear_var = create_combo_box(gui, 4, "Activation Range Rear:", [0, 1])
    is_battery_var = create_combo_box(gui, 5, "Is Battery Low:", [0, 1])
    robot_state_var = create_combo_box(gui, 6, "Robot State:", ['moving', 'stopped'])
    ur5_state_var = create_combo_box(gui, 7, "UR5 State:", ['moving', 'stopped'])

    transport_request_var = create_text_field(gui, 8, "Transport Requests:")    
    
    goal_intention_var = create_combo_box(gui, 9, "Goal Intention:", ['EXPLORATION', 'CHARGE', 'TRANSPORT'])
    goal_pose_var = create_text_field(gui, 10, "Goal Pose:")    
    
    create_button(gui, 11, 0, "Load Planner Model File", open_model_file)
    create_button(gui, 11, 1, "Load Graph File", open_json_file)

    create_button(gui, 12, 0, "Load Dictionary File", open_dictionary_json_file)

    clear_button = create_button(gui, 13, 0, "Clear Board", clear_board, 'disabled')
    check_button = create_button(gui, 13, 1, "Check Plan", check_state_plan, 'disabled')

    gui.mainloop()

    