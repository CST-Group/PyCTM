import unittest

from pyctm.representation.idea import Idea
from pyctm.representation.sdr_idea_serializer import SDRIdeaSerializer


class SDRIdeaSerializerTest(unittest.TestCase):

    def init_idea(self):

        idea = Idea(0, "Rock Music", "Hey ho let's go!", 0);
        idea.add(Idea(1, "Metallica", "Black Album", 0)).add(Idea(2, "Unforgiven", 3.14, 1)).add(Idea(3,"Enter Sadman", "Seek and destroy"))
        idea.add(Idea(4, "Foo Fighters", "The sky's the neighborhood", 0)).add(Idea(5, "Pretender", 256))
        idea.add(Idea(6, "Black Sabbath", [3.41, 2.22, 0.23], 1)).add(Idea(7, "Paranoid", [34, 18, 10]));
        idea.add(Idea(8, "Gun's in Roses", "Sweet child o' mine", 2)).add(Idea(9, "November Rain", [-18, 1.2, 2, 5.2, -1, 0, 1000]));
    
        return idea

    def test_sdr_serialization(self):

        sdr_idea_serializer = SDRIdeaSerializer(16, 32, 32)

        idea = self.init_idea()

        matrix_idea = sdr_idea_serializer.serialize(idea)

        print(matrix_idea)

        


if __name__ == '__main__':
    unittest.main()