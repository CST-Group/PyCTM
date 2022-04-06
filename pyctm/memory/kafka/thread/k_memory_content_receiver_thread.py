import json
from threading import Thread, Event
import time


class KMemoryContentReceiverThread(Thread):

    def __init__(self, memory=None, consumer=None, topic_config=None):
        super(KMemoryContentReceiverThread, self).__init__()
        self.memory = memory
        self.consumer = consumer
        self.topic_config = topic_config
        self._stop_event = Event()
        self.stopped = False

    def stop(self):
        self.stopped = True
        self._stop_event.set()

    def is_stop(self):
        self._stop_event.is_set()

    def run(self):

        while not self.stopped:
            message = self.consumer.poll(10)

            if message is None:
                continue
            if message.error():
                print('Consumer error: %s' % message.error())
                continue

            j = json.loads(message.value().decode('utf-8'))

            self.memory.set_evaluation(j["evaluation"])
            self.memory.set_i(j["i"])

            time.sleep(0.01)
