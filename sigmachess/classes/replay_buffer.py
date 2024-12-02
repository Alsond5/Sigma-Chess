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

    def augment_data(self):
        augmented_buffer = []
        for state, policy, value in self.buffer:
            # Original sample
            augmented_buffer.append((state, policy, value))

            # 1D policy vektörünü 2D hale getirme
            policy_2d = policy.reshape((4672, 1))  # (4672,) -> (4672, 1)

            # Apply augmentation: Rotate 90 degrees (clockwise)
            rotated_state = np.rot90(state)
            rotated_policy = np.rot90(policy_2d)
            augmented_buffer.append((rotated_state, rotated_policy, value))

            # Apply augmentation: Flip horizontally
            flipped_state = np.flip(state, axis=1)
            flipped_policy = np.flip(policy, axis=1)
            augmented_buffer.append((flipped_state, flipped_policy, value))

            # Apply augmentation: Flip vertically
            flipped_vertical_state = np.flip(state, axis=0)
            flipped_vertical_policy = np.flip(policy, axis=0)
            augmented_buffer.append((flipped_vertical_state, flipped_vertical_policy, value))

        # Replace original buffer with augmented buffer
        self.buffer = deque(augmented_buffer, maxlen=self.buffer.maxlen)