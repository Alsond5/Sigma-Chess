import math
import numpy as np
import tensorflow as tf

class Node:
    def __init__(self, state, parent=None, prior_prob=1.0):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value_sum = 0
        self.prior_prob = prior_prob
        self.is_expanded = False

    @property
    def value(self):
        return self.value_sum / (self.visits + 1e-5)

    def expand(self, action_probs):
        for action, prob in enumerate(action_probs):
            if prob > 0:
                next_state = self.state.clone()
                next_state.get_next_state(action)
                self.children[action] = Node(next_state, parent=self, prior_prob=prob)
        self.is_expanded = True

    def select(self, c_puct=1.0):
        max_ucb = -float('inf')
        best_action = None
        best_child = None
        
        for action, child in self.children.items():
            ucb = child.value + c_puct * child.prior_prob * (math.sqrt(self.visits) / (1 + child.visits))
            if ucb > max_ucb:
                max_ucb = ucb
                best_action = action
                best_child = child
        return best_action, best_child

    def backup(self, value):
        self.visits += 1
        self.value_sum += value
        if self.parent:
            self.parent.backup(-value)

class MCTS:
    def __init__(self, model, c_puct=1.0, simulations=50):
        self.model = model
        self.c_puct = c_puct
        self.simulations = simulations

    def add_dirichlet_noise(self, node, valid_moves):
        noise = np.random.dirichlet([0.3] * len(valid_moves))
        for idx, action in enumerate(valid_moves):
            if action in node.children:
                node.children[action].prior_prob = \
                    0.75 * node.children[action].prior_prob + 0.25 * noise[idx]

    def run(self, initial_state, temperature=1.0):
        root = Node(initial_state)
        
        # First evaluate and expand root
        action_probs, value = self.evaluate(initial_state)
        valid_moves = initial_state.get_valid_moves()
        
        # Add Dirichlet noise to root (alpha=0.3 for chess)
        noise = np.random.dirichlet([0.3] * len(valid_moves))

        # Expand with noisy priors
        for idx, action in enumerate(valid_moves):
            prob = action_probs[action]
            noisy_prob = 0.75 * prob + 0.25 * noise[idx]
            next_state = initial_state.clone()
            next_state.get_next_state(action)
            root.children[action] = Node(next_state, parent=root, prior_prob=noisy_prob)

        for _ in range(self.simulations):
            node = root
            
            # Selection
            while node.is_expanded and not node.state.is_terminal():
                action, node = node.select(self.c_puct)
            
            # Expansion and Evaluation
            if not node.state.is_terminal():
                action_probs, value = self.evaluate(node.state)
                valid_moves = node.state.get_valid_moves()
                node.expand(action_probs)
            else:
                value = node.state.get_winner()
                value = 1 if value == node.state.player_color else (0 if value == 2 else -1)
            
            # Backup
            node.backup(value)

        return self.get_action_probs(root, temperature)

    def evaluate(self, state):
        state_tensor = state.get_current_state()
        # state_tensor = np.expand_dims(state_tensor, axis=0)
        
        policy, value = self.model.predict(state_tensor, verbose=0)
        
        # Mask invalid moves
        valid_moves = state.get_valid_moves()
        mask = np.zeros(policy.shape[1])
        mask[valid_moves] = 1
        
        policy = policy[0] * mask
        
        # Normalize
        sum_policy = np.sum(policy)
        if sum_policy > 0:
            policy /= sum_policy
        else:
            # If all moves were masked, use uniform distribution over valid moves
            policy = mask / np.sum(mask)
            
        return policy, value[0][0]

    def get_action_probs(self, root, temperature=1.0):
        visits = np.array([child.visits for action, child in root.children.items()])
        actions = list(root.children.keys())
        
        if temperature == 0:  # Pure exploitation
            action_idx = np.argmax(visits)
            probs = np.zeros_like(visits)
            probs[action_idx] = 1
        else:
            # Apply temperature
            visits = visits ** (1 / temperature)
            probs = visits / np.sum(visits)
        
        # Convert to full move probability vector
        full_probs = np.zeros(4672)  # Adjust size based on your action space
        for action, prob in zip(actions, probs):
            full_probs[action] = prob
            
        return full_probs