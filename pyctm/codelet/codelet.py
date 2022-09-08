from threading import Thread

import time


class Codelet(Thread):
    def __init__(self, name):
        Thread.__init__()
        self.activation = 0
        self.threshold = 0
        self.name = name
        self.enabled = True
        self.timestamp = 300

        self.inputs = []
        self.outputs = []
        self.broadcast = []

    def proc(self):
        pass

    def calculate_activation(self):
        pass

    def access_memory_objects(self):
        pass

    def run(self):

        while self.enabled:
            self.access_memory_objects()
            self.calculate_activation()
            self.proc()
            time.sleep(self.timestamp)

    def add_input(self, input):
        self.inputs.append(input)

    def add_output(self, output):
        self.outputs.append(output)

    def get_activation(self):
        return self.activation

    def set_activation(self, activation):

        if activation > 1:
            self.activation = 1
        elif activation < 0:
            self.activation = 0
        else:
            self.activation = activation

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs

    def get_output(self, name):
        for output in self.outputs:
            if output.get_name() == name:
                return output
        
        return None

    def get_input(self, name):
        for input_m in self.inputs:
            if input_m.get_name() == name:
                return input_m
        
        return None

    def get_broadcast(self):
        return self.broadcast

    def get_broadcast(self, name):
        for broadcast_m in self.broadcast:
            if broadcast_m.get_name() == name:
                return broadcast_m
        
        return None
