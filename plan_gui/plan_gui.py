from tkinter import *
from tkinter import ttk
from tkinter.filedialog import askopenfile 
from matplotlib import pyplot as plt, patches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json

def open_and_draw_graph_window(gui=None):
    
    global new_window
    global ax

    new_window = Toplevel(gui)
 
    new_window.title("Graph Map")
    new_window.geometry("800x800")

    figure = plt.Figure(figsize=(20, 20), dpi=100)
    ax = figure.add_subplot()
    ax.set_xlim([-15, 15])
    ax.set_ylim([-15, 15])

    for node in graph['nodes']:
        draw_arrow(ax, node, graph['nodes'])
    
    for node in graph['nodes']:
        draw_node(ax, node)        

    figure_canvas = FigureCanvasTkAgg(figure, new_window)
    figure_canvas.get_tk_widget().pack(side=LEFT, fill=BOTH)

def open_json_file(gui=None):
    global graph

    file_path = askopenfile(mode='r', filetypes=[('Json File', '.json')])
    if file_path is not None:
        graph = json.load(file_path)
        clear_button['state'] = 'normal'
        open_and_draw_graph_window(gui)

def open_model_file(gui=None):
    pass

def draw_arrow(ax, start_node, nodes):
    start_coordinates = start_node['coordinates']

    for connection in start_node['connected']:
        end_node = nodes[connection-1]
        end_coordinates = end_node['coordinates']
        arrow = patches.FancyArrow(start_coordinates[0], start_coordinates[1], end_coordinates[0]-start_coordinates[0], end_coordinates[1]-start_coordinates[1])
        ax.add_patch(arrow)

def draw_node(ax, node):
    coordinates = node['coordinates']
    circle = patches.Circle((coordinates[0], coordinates[1]), radius=1, color='gold')
    ax.add_patch(circle)
    ax.annotate(node['id'], xy=(coordinates[0], coordinates[1]), fontsize=12, ha="center")


def clear_board(gui=None):

    if new_window is not None:
        new_window.destroy()
    
    open_and_draw_graph_window(gui)

def create_gui():
    gui = Tk()
    gui.title("Plan Generated Test.")
    gui.geometry("400x320")

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

def create_button(gui, row, column, text, func=None, state=None):
    button = Button(gui, text=text, command=lambda:func(gui) if func else None, state=state if state else 'normal')
    button.grid(row=row, column=column, pady=3)

    return button

if __name__ == '__main__':
    gui = create_gui()
    
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

    global clear_button
    
    create_button(gui, 11, 0, "Load Planner Model File", open_model_file)
    create_button(gui, 11, 1, "Load Graph File", open_json_file)

    clear_button = create_button(gui, 12, 0, "Clear Board", clear_board, 'disabled')
    create_button(gui, 12, 1, "Check Plan")
    

    gui.mainloop()

    