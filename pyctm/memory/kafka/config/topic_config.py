from pyctm.memory.kafka.k_distributed_memory_behavior import KDistributedMemoryBehavior


class TopicConfig:

    def __init__(self, broker, name, k_distributed_memory_behavior=KDistributedMemoryBehavior.PULLED,
                 regex_pattern=None, class_name=None) -> None:
        self.name = name
        self.k_distributed_memory_behavior = k_distributed_memory_behavior
        self.class_name = class_name
        self.broker = broker
        self.regex_pattern = regex_pattern
