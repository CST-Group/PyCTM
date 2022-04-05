import json
from threading import Thread
import time

from pyctm.memory.kafka.builder.k_producer_builder import KProducerBuilder
from pyctm.memory.kafka.k_distributed_memory_behavior import KDistributedMemoryBehavior


class KMemoryContentPublisherThread(Thread):

    def __init__(self, memory=None, producer=None, topic_config=None):
        Thread.__init__(self)
        self.memory = memory
        self.producer = producer
        self.topic_config = topic_config
        self.last_i = None
        self.last_evaluation = 0

        KProducerBuilder.check_topic_exist(topic_config.broker, topic_config.name)

    def run(self):

        print('Content publisher thread initialized for memory %s.' % self.memory.get_name())

        while True:
            if self.topic_config.k_distributed_memory_behavior == KDistributedMemoryBehavior.TRIGGERED:
                self.memory.locked = True
                while self.memory.locked:
                    continue

                object_json = json.dumps(self.memory.__dict__)

                self.producer.poll(10)
                self.producer.produce(self.topic_config.name, object_json)
            else:

                if self.memory.get_i() != self.last_i or self.memory.get_evaluation() != self.last_evaluation:
                    object_json = json.dumps(self.memory.__dict__)

                    self.producer.poll(10)
                    self.producer.produce(self.topic_config.name, object_json)

                    self.last_evaluation = self.memory.get_evaluation()
                    self.last_i = self.memory.get_i()

            time.sleep(0.01)
