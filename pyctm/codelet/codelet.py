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

    def run(self):

        while self.enabled:
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

    def get_broadcast(self):
        return self.broadcast
