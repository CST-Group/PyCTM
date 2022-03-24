

import unittest
from pyctm.memory import memory_object

# from pyctm.memory.memory import MemoryObject
from pyctm.memory.memory_object import MemoryObject

class MemoryTest(unittest.TestCase):

    def test_memory_initialization(self):

        memoryObject = MemoryObject(1, "test", 10, 1)

        assert memoryObject.get_id() == 1
        assert memoryObject.get_name() == "test"
        assert memoryObject.get_i() == 10
        assert memoryObject.get_evaluation() == 1

    def test_memory_set_evaluation(self):   

        memoryObject = MemoryObject(1, "test", 10, 10)
        assert memoryObject.get_evaluation() == 1

        memoryObject.set_evaluation(-10)
        assert memoryObject.get_evaluation() == 0

if __name__ == '__main__':
    unittest.main()