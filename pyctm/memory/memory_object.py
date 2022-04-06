from pyctm.memory.memory import Memory


class MemoryObject(Memory):

    def __init__(self, id=0, name="", i=None, evaluation=0):
        self.id = id
        self.name = name
        self.i = i
        self.locked = False
        self.set_evaluation(evaluation)

    def get_i(self):
        return self.i

    def get_name(self):
        return self.name

    def get_evaluation(self):
        return self.evaluation

    def get_id(self):
        return self.id

    def set_i(self, i):
        self.i = i

        if self.locked:
            self.locked = False

    def set_name(self, name):
        self.name = name

    def set_evaluation(self, evaluation):
        if evaluation > 1:
            self.evaluation = 1
        elif evaluation < 0:
            self.evaluation = 0
        else:
            self.evaluation = evaluation

    def set_id(self, id):
        self.id = id
