import unittest
import time
from pyctm.memory.distributed_memory_type import DistributedMemoryType


from pyctm.memory.k_distributed_memory import KDistributedMemory
from pyctm.memory.kafka.config.topic_config import TopicConfig
from pyctm.memory.kafka.k_distributed_memory_behavior import KDistributedMemoryBehavior


class KDistributedMemoryTest(unittest.TestCase):

    @staticmethod
    def memory_initialization():

        input_topic_configs = []
        output_topic_configs = []

        input_topic_configs.append(TopicConfig("localhost:9092", "topic-1", KDistributedMemoryBehavior.PULLED))
        output_topic_configs.append(TopicConfig("localhost:9092", "topic-1", KDistributedMemoryBehavior.TRIGGERED))

        distributed_output_memory = KDistributedMemory("MEMORY_TEST_OUTPUT", DistributedMemoryType.OUTPUT_MEMORY,
                                                       output_topic_configs)
        distributed_input_memory = KDistributedMemory("MEMORY_TEST_INPUT", DistributedMemoryType.INPUT_MEMORY,
                                                      input_topic_configs)

        return distributed_input_memory, distributed_output_memory

    def test_should_has_message_in_input_distributed_memory(self):
        distributed_input_memory, distributed_output_memory = self.memory_initialization()

        message = "Test message in the distributed memory!"

        distributed_output_memory.set_i(message)

        time.sleep(2)
        assert distributed_output_memory.get_i() == distributed_input_memory.get_i()

        message = "New message to test!"
        distributed_output_memory.set_i(message)

        time.sleep(2)
        assert distributed_output_memory.get_i() == distributed_input_memory.get_i()

        distributed_output_memory.stop()
        distributed_input_memory.stop()

if __name__ == '__main__':
    unittest.main()