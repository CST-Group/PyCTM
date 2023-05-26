import unittest

import json

from pyctm.representation.idea import Idea
from pyctm.representation.sdr_idea_deserializer import SDRIdeaDeserializer
from pyctm.representation.sdr_idea_serializer import SDRIdeaSerializer
from pyctm.representation.dictionary import Dictionary



class SDRIdeaSerializerTest(unittest.TestCase):

    def init_idea(self):

        idea = Idea(0, "Rock Music", "Hey ho let's go!", 0);
        idea.add(Idea(1, "Metallica", "Black Album", 0)).add(Idea(10, "Unforgiven", 3.14, 1)).add(Idea(3,"Enter Sadman", "Seek and destroy"))
        idea.add(Idea(4, "Foo Fighters", "The sky's the neighborhood", 0)).add(Idea(5, "Pretender", 256))
        idea.add(Idea(6, "Black Sabbath", [3.41, 2.22, 0.23], 1)).add(Idea(7, "Paranoid", [34, 18, 10]));
        idea.add(Idea(8, "Gun's in Roses", "Sweet child o' mine", 2)).add(Idea(9, "November Rain", [-18, 1.2, 2, 5.2, -1, 0, 1000]));
    
        return idea

    #def test_sdr_serialization(self):
    #    
    #    file = open("/opt/repository/dataTrainingShortSDR/dictionary.json")
#
    #    object=json.load(file)
    #    dictionary = Dictionary(**object)
#
    #    sdr_idea_serializer = SDRIdeaSerializer(16, 32, 32)
    #    sdr_idea_serializer.dictionary = dictionary
#
    #    idea = self.init_idea()
#
    #    sdr_idea = sdr_idea_serializer.serialize(idea)
#
    #    print(sdr_idea)
#
    #    sdr_idea_deserializer = SDRIdeaDeserializer(sdr_idea_serializer.dictionary)
#
    #    converted_idea = sdr_idea_deserializer.deserialize(sdr_idea.sdr)
#
    #    print(converted_idea)

    

    def test_sdr_compability(self):

        file = open("/opt/repository/dataTrainingShortSDR/dictionary.json")

        object=json.load(file)
        dictionary = Dictionary(**object)

        sdr_idea_serializer = SDRIdeaSerializer(10, 32, 32)
        sdr_idea_serializer.dictionary = dictionary


        goal_idea = Idea(_id=0, name="Goal", value="", _type=1)
        init_pose_idea = Idea(_id=1, name="initNodeId", value=1.0, _type=1)
        middle_pose_idea = Idea(_id=2, name="middleTagId", value=403.0, _type=1)
        goal_pose_idea = Idea(_id=3, name="goalTagId", value=62.0, _type=1)
        context_idea = Idea(_id=4, name="context", value="", _type=0)
        
        goal_idea.add(init_pose_idea)
        goal_idea.add(middle_pose_idea)
        goal_idea.add(goal_pose_idea)
    
        context_idea.add(Idea(_id=5, name='idle', value="", _type=1))
        context_idea.add(Idea(_id=6, name='moveToNode', value=1.0, _type=1))
        context_idea.add(Idea(_id=7, name='moveToNode', value=16.0, _type=1))
        #context_idea.add(Idea(_id=8, name='moveToNode', value=15.0, _type=1))
        
        goal_idea.add(context_idea)

        
        goal_idea_sdr = sdr_idea_serializer.serialize(goal_idea)

        file = open("/opt/repository/PyCTM/pyctm/resources/dataTrainingShortSDR_test.json")
        object=json.load(file)

        goal_sdr_target = object["x"]

        print(self.compare_sdr(goal_idea_sdr.sdr, goal_sdr_target))


    


if __name__ == '__main__':
    unittest.main()