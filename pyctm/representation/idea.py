
class Idea():
    
    def __init__(self, _id=0, name = "", value=None, _type = 0):
        self.name = name
        self.value = value
        self.id = _id
        self.type = _type
        self.last_id = 0
        self.child_ideas = []

    def add(self, idea):
        self.child_ideas.append(idea)
        return idea