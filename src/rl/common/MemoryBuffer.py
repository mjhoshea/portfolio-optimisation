import random
from collections import deque


class MemoryBuffer:

    def __init__(self, buffer_size):
        self.capacity = buffer_size
        self.memory = deque(maxlen=buffer_size)

    def __len__(self):
        return len(self.memory)

    def add(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def clear(self):
        self.memory = deque(maxlen=self.capacity)
