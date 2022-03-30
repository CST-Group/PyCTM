import enum


class DistributedMemoryType(enum.Enum):
    OUTPUT_MEMORY = 0,
    INPUT_MEMORY = 1,
    INPUT_BROADCAST_MEMORY = 2,
    OUTPUT_BROADCAST_MEMORY = 3
