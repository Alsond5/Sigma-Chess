import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, maxlen=100000):
        self.buffer = deque(maxlen=maxlen)
        self.state_shape = (8, 8, 119)  # Expected chess state shape

    def store(self, state, policy, value):
        # Ensure state has correct shape
        if state.shape != self.state_shape:
            state = np.reshape(state, self.state_shape)
        self.buffer.append((state, policy, value))

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
            
        indices = np.random.choice(len(self.buffer), size=batch_size)
        states, policies, values = [], [], []
        
        for idx in indices:
            state, policy, value = self.buffer[idx]
            # Ensure consistent shapes
            state = np.reshape(state, self.state_shape)
            states.append(state)
            policies.append(policy)
            values.append(value)
            
        return (np.array(states, dtype=np.float32), 
                np.array(policies, dtype=np.float32), 
                np.array(values, dtype=np.float32))

    def augment_data(self):
        augmented_buffer = []
        
        for state, policy, value in self.buffer:
            # Ensure state has correct shape
            state = np.reshape(state, self.state_shape)
            policy_3d = policy.reshape(8, 8, -1)
            
            # Generate all 8 orientations
            for rot in range(4):
                for flip in [False, True]:
                    # Rotate
                    new_state = np.rot90(state, k=rot, axes=(0, 1))
                    new_policy = np.rot90(policy_3d, k=rot, axes=(0, 1))
                    
                    # Flip
                    if flip:
                        new_state = np.flip(new_state, axis=1)
                        new_policy = np.flip(new_policy, axis=1)
                    
                    # Reshape policy back to 1D
                    new_policy = new_policy.reshape(-1)
                    
                    # Ensure state shape consistency
                    new_state = np.reshape(new_state, self.state_shape)
                    
                    augmented_buffer.append((new_state, new_policy, value))
        
        self.buffer = deque(augmented_buffer, maxlen=self.buffer.maxlen)