import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, maxlen=100000) -> None:
        self.buffer = deque(maxlen=maxlen)

    def store(self, state, policy, value):
        self.buffer.append((state, policy, value))

    def sample(self, batch_size):
        buffer_array = np.array(self.buffer, dtype=object)  # Use dtype=object for variable shapes
        indices = np.random.choice(len(self.buffer), size=batch_size)
        batch = buffer_array[indices]
        states, policies, values = zip(*batch)  # Unzip into separate lists or arrays

        return np.array(states), np.array(policies), np.array(values)