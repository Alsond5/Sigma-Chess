import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, maxlen=500000):
        self.buffer = deque(maxlen=maxlen)
        self.current_size = 0

    def store(self, state, policy, value):
        """Store a single game state transition"""
        self.buffer.append({
            'state': state,
            'policy': policy,
            'value': value
        })
        self.current_size = len(self.buffer)

    def sample(self, batch_size):
        """Sample a batch with augmentations"""
        if self.current_size < batch_size:
            batch_size = self.current_size

        indices = np.random.choice(self.current_size, batch_size)
        states, policies, values = [], [], []

        for idx in indices:
            sample = self.buffer[idx]
            # Get augmented samples
            aug_states, aug_policies = self._augment_sample(
                sample['state'], 
                sample['policy']
            )
            
            # Add all augmentations
            states.extend(aug_states)
            policies.extend(aug_policies)
            values.extend([sample['value']] * len(aug_states))

        return np.array(states), np.array(policies), np.array(values)

    def _augment_sample(self, state, policy):
        """Generate valid augmentations for a single sample"""
        # Remove batch dimension if present
        if len(state.shape) == 4:
            state = np.squeeze(state, axis=0)
        
        augmented_states = [state]
        augmented_policies = [policy]

        # Horizontal flip
        flip_h = np.flip(state, axis=1)
        augmented_states.append(flip_h)
        augmented_policies.append(policy)  # Policy needs game-specific mapping

        # Vertical flip 
        flip_v = np.flip(state, axis=0)
        augmented_states.append(flip_v)
        augmented_policies.append(policy)  # Policy needs game-specific mapping

        # Diagonal flip (only if shape allows)
        if state.shape[0] == state.shape[1]:
            diag = np.transpose(state, (1, 0, 2))
            augmented_states.append(diag)
            augmented_policies.append(policy)  # Policy needs game-specific mapping

        return augmented_states, augmented_policies

    def __len__(self):
        return self.current_size