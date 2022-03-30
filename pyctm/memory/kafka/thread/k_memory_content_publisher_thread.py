
import json
from threading import Thread
import time

from pyctm.memory.kafka.k_distributed_memory_behavior import KDistributedMemoryBehavior


class KMemoryContentPublisherThread(Thread):

    def __init__(self, memory=None, producer=None, topic_config=None):
        self.memory = memory
        self.producer = producer
        self.topic_config = topic_config
        self.last_i = None
        self.last_evaluation = 0

    def run(self):

        while True:
            if self.topic_config.distributed_memory_type == KDistributedMemoryBehavior.TRIGGERED:               
                
                object_json = json.dump(self.memory.__dict__)

                self.producer.poll(10)
                self.producer.produce(self.topic_config.name, object_json)
            else:

                if self.memory.get_i() != self.last_i or self.memory.get_evaluation() != self.last_evaluation:
                    object_json = json.dump(self.memory.__dict__)

                    self.producer.poll(10)
                    self.producer.produce(self.topic_config.name, object_json)
                    
                    self.last_evaluation = self.memory.get_evaluation()
                    self.last_i = self.memory.get_i()

            time.sleep(0.01)