import json
from threading import Thread, Event
import time

from pyctm.memory.kafka.builder.k_producer_builder import KProducerBuilder
from pyctm.memory.kafka.k_distributed_memory_behavior import KDistributedMemoryBehavior


class KMemoryContentPublisherThread(Thread):

    def __init__(self, memory=None, producer=None, topic_config=None, condition=None):
        super(KMemoryContentPublisherThread, self).__init__()
        self.memory = memory
        self.producer = producer
        self.topic_config = topic_config
        self.last_i = None
        self.last_evaluation = 0
        self.condition = condition
        self._stop_event = Event()
        self.stopped = False

        KProducerBuilder.check_topic_exist(topic_config.broker, topic_config.name)

    def stop(self):
        self.stopped = True
        self.condition.acquire()
        self.condition.notify()
        self._stop_event.set()

    def is_stop(self):
        return self._stop_event.is_set()

    def run(self):

        print('Content publisher thread initialized for memory %s.' % self.memory.get_name())

        while not self.stopped:
            if self.topic_config.k_distributed_memory_behavior == KDistributedMemoryBehavior.TRIGGERED:
                self.condition.acquire()
                self.condition.wait()

                object_json = json.dumps(self.memory.__dict__)

                self.producer.poll(10)
                self.producer.produce(self.topic_config.name, object_json)

                self.condition.release()
            else:

                if self.memory.get_i() != self.last_i or self.memory.get_evaluation() != self.last_evaluation:
                    object_json = json.dumps(self.memory.__dict__)

                    self.producer.poll(10)
                    self.producer.produce(self.topic_config.name, object_json)

                    self.last_evaluation = self.memory.get_evaluation()
                    self.last_i = self.memory.get_i()

            time.sleep(0.01)

