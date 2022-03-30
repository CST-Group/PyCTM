

from pyctm.memory.distributed_memory_type import DistributedMemoryType
from pyctm.memory.kafka.builder.k_producer_builder import KProducerBuilder
from pyctm.memory.memory import Memory
from pyctm.memory.kafka.builder.k_consumer_builder import KConsumerBuilder
from pyctm.memory.memory_object import MemoryObject
from pyctm.memory.kafka.thread.k_memory_content_receiver_thread import KMemoryContentReceiverThread


class KDistributedMemory(Memory):

    def __init__(self, name="", distributed_memory_type=DistributedMemoryType.INPUT_MEMORY, topics_config=[]):

        self.memory_setup(name, distributed_memory_type, topics_config)

    def memory_setup(self, name, distributed_memory_type, topics_config):

        self.name = name
        self.distributed_memory_type = distributed_memory_type
        self.topics_config = topics_config

        print('Creating KDistributedMemory %s for type %s.' %
              (name, distributed_memory_type))

        self.memories = []
        self.k_memory_content_receiver_threads = []
        self.k_memory_content_publisher_threads = []

        self.init_memory()

        print('KDistributeMemory %s created.' % name)

    def init_memory(self):

        if self.distributed_memory_type == 'INPUT_MEMORY' or self.distributed_memory_type == 'BROADCAST_MEMORY':
            self.consumers_setup(self.topics_config)
        else:
            self.producers_setup(self.topics_config)

    def consumers_setup(self, topics_config):
        print('Creating the consumers.')

        topics_consumer_map = KConsumerBuilder.generate_consumers(
            topics_config, self.name)

        for topic_config, consumer in topics_consumer_map.items():

            memory = MemoryObject(0, topic_config.name)
            self.memories.append(memory)

            k_memory_content_receiver_thread = KMemoryContentReceiverThread(
                memory, consumer, topics_config)
            k_memory_content_receiver_thread.start()

            self.k_memory_content_receiver_threads.append(
                k_memory_content_receiver_thread)

        print('Consumers created.')

    def producers_setup(self, topics_config):
        print('Creating the producers.')

        producers = KProducerBuilder.generateProducers(topics_config)

        for (producer, topic_config) in zip(producers, topics_config):
            memory = MemoryObject(0, topic_config.name)
            self.memories.append(memory)

            k_memory_content_publisher_thread = KMemoryContentPublisherThread(memory, producer, topic_config)
            k_memory_content_publisher_thread.start()

            self.k_memory_content_publisher_threads.append(k_memory_content_publisher_thread)

        print('Producers created.')


