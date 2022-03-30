import json
from threading import Thread
from pyctm.memory.memory import Memory
import time


class KMemoryContentReceiverThread(Thread):

    def __init__(self, memory=None, consumer=None, topic_config=None):
        self.memory = memory
        self.consumer = consumer
        self.topic_config = topic_config

    def run(self):

        while True:
            message = self.consumer.poll(10)

            if message is None:
                continue
            if message.error():
                print('Consumer error: %s' % message.error())

            j = json.loads(message.value().decode('utf-8'))
            memory = Memory(**j)

            self.memory.set_evaluation(memory.get_evaluation())
            self.memory.set_i(memory.get_i())

            time.sleep(0.01)
