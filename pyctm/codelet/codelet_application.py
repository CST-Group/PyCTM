from pyctm.memory.distributed_memory_type import DistributedMemoryType
from pyctm.memory.k_distributed_memory import KDistributedMemory


class CodeletApplication():

    def __init__(self, codelet_memories):
        self.codelet_memories = codelet_memories
        self.codelets = []

        self.initialize_distributed_memories()

    def initialize_distributed_memories(self):

        print('Inserting codelets into Codelet Application.')

        for codelet, memories in self.codelet_memories.items():
            for memory in memories:

                if isinstance(memory, KDistributedMemory):

                    if memory.distributed_memory_type == DistributedMemoryType.INPUT_MEMORY:
                        if len(list(filter(lambda input_memory: (input_memory == memory), codelet.inputs))) > 0:
                            codelet.inputs.append(memory)

                    elif memory.distributed_memory_type == DistributedMemoryType.OUTPUT_MEMORY:
                        if len(list(filter(lambda output_memory: (output_memory == memory), codelet.outputs))) > 0:
                            codelet.outputs.append(memory)
                    else:
                        if len(list(filter(lambda broadcast_memory: (broadcast_memory == memory), codelet.broadcast))) > 0:
                            codelet.broadcast.append(memory)

            self.codelets.append(codelet)

            print('Starting codelet %s' % codelet.name)
            codelet.start()
            print('Codelet %s started.' % codelet.name)


