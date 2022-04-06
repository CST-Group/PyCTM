import threading

from pyctm.memory.distributed_memory_type import DistributedMemoryType
from pyctm.memory.kafka.builder.k_producer_builder import KProducerBuilder
from pyctm.memory.kafka.thread.k_memory_content_publisher_thread import KMemoryContentPublisherThread
from pyctm.memory.memory import Memory
from pyctm.memory.kafka.builder.k_consumer_builder import KConsumerBuilder
from pyctm.memory.memory_object import MemoryObject
from pyctm.memory.kafka.thread.k_memory_content_receiver_thread import KMemoryContentReceiverThread


class KDistributedMemory(Memory):

    def __init__(self, name="", distributed_memory_type=DistributedMemoryType.INPUT_MEMORY, topics_config=[]):
        self.name = name
        self.distributed_memory_type = distributed_memory_type
        self.topics_config = topics_config

        print('Creating KDistributedMemory %s for type %s.' %
              (name, distributed_memory_type))

        self.memories = []
        self.k_memory_content_receiver_threads = []
        self.k_memory_content_publisher_threads = []

        self.condition = threading.Condition()

        self.init_memory()

        print('KDistributeMemory %s created.' % name)

    def init_memory(self):

        if self.distributed_memory_type == DistributedMemoryType.INPUT_MEMORY \
                or self.distributed_memory_type == DistributedMemoryType.INPUT_BROADCAST_MEMORY:
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

        producers = KProducerBuilder.generate_producers(topics_config)

        for (producer, topic_config) in zip(producers, topics_config):
            memory = MemoryObject(0, topic_config.name)
            self.memories.append(memory)

            k_memory_content_publisher_thread = KMemoryContentPublisherThread(memory, producer, topic_config,
                                                                              self.condition)
            k_memory_content_publisher_thread.start()

            self.k_memory_content_publisher_threads.append(k_memory_content_publisher_thread)

        print('Producers created.')

    def stop(self):
        for k_thread in self.k_memory_content_receiver_threads:
            k_thread.stop()

        for k_thread in self.k_memory_content_publisher_threads:
            k_thread.stop()


    def get_i(self):
        memory = max(self.memories, key=lambda m: m.get_i())
        return memory.get_evaluation()

    def get_i_index(self, index):
        try:
            return self.memories[index]
        except IndexError:
            print('Impossible to get memory content. Index %s out of bounds.' % index)
            return None

    def get_memory(self):
        return max(self.memories, key=lambda memory: memory.get_i())

    def set_i(self, i):
        try:
            self.condition.acquire()
            self.memories[0].set_i(i)
            self.condition.notify()
        except IndexError:
            print('Impossible to set memory content. Index 0 out of bounds.')
        finally:
            self.condition.release()

    def set_i_index(self, i, index):
        try:
            self.condition.acquire()
            self.memories[index].set_i(i)
            self.condition.notify()
        except IndexError:
            print('Impossible to get memory content. Index %s out of bounds.' % index)
        finally:
            self.condition.release()

    def set_i_evaluation_index(self, i, evaluation, index):
        try:
            self.condition.acquire()
            self.memories[index].set_i(i)
            self.memories[index].set_evaluation(evaluation)
            self.condition.notify()
        except IndexError:
            print('Impossible to get memory content. Index %s out of bounds.' % index)
        finally:
            self.condition.release()

    def get_evaluation(self):
        memory = max(self.memories, key=lambda m: m.get_i())
        if memory is not None:
            return memory.get_evaluation()

        return -1

    def set_evaluation(self, evaluation):
        try:
            self.condition.acquire()
            self.memories[0].set_evaluation(evaluation)
            self.condition.notify()
        except IndexError:
            print('Impossible to get memory content. Index 0 out of bounds.')
        finally:
            self.condition.release()

    def set_evaluation_index(self, evaluation, index):
        try:
            self.condition.acquire()
            self.memories[index].set_evaluation(evaluation)
            self.condition.notify()
        except IndexError:
            print('Impossible to get memory content. Index %s out of bounds.' % index)
        finally:
            self.condition.release()

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name
