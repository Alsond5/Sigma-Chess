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
            if prob <= 0:
                continue
            
            next_state = self.state.clone()
            next_state.get_next_state(action)
            child = Node(next_state, parent=self, prior_prob=prob)
            self.children[action] = child

        self.is_expanded = True

    def select(self, c_puct=1.0):
        max_ucb = -float('inf')
        best_action = None
        best_child = None
        for action, child in self.children.items():
            ucb = child.value + c_puct * child.prior_prob * (math.sqrt(self.visits + 1e-5) / (1 + child.visits))
            if ucb > max_ucb:
                max_ucb = ucb
                best_action = action
                best_child = child
        return best_action, best_child

    def backup(self, value):
        self.visits += 1
        self.value_sum += value

        if self.parent:
            self.parent.backup(-value)  # Flip the value for the opponent's turn

class MCTS:
    def __init__(self, model, c_puct=1.0, simulations=50):
        self.model = model
        self.c_puct = c_puct
        self.simulations = simulations

    def run(self, initial_state):
        root = Node(initial_state)

        for _ in range(self.simulations):
            node = root
            
            # Selection
            while node.is_expanded:
                action, node = node.select(self.c_puct)
            
            # Expansion and Evaluation
            leaf_state = node.state

            if not leaf_state.is_terminal():
                action_probs, value = self.evaluate(leaf_state)
                node.expand(action_probs)
            else:
                value = 1 if leaf_state.get_winner() == 0 else -1
            
            # Backup
            node.backup(value)

        return self.get_best_action(root)

    def evaluate(self, state):
        state_tensor = state.get_current_state()
        output, value = self.model.predict(state_tensor, verbose=0)
        legal_moves = state.get_valid_moves()

        if len(legal_moves) == 0:
            return np.zeros_like(output), value[0]
        
        output = output[0]
        value = value[0]

        legal_moves_mask = np.zeros_like(output)
        legal_moves_mask[legal_moves] = 1

        output *= legal_moves_mask
        total = np.sum(output)

        if total == 0:
            output[legal_moves] = 1 / len(legal_moves)
        else:
            output /= total

        return output, value[0]

    def get_best_action(self, root):
        total_visits = sum(child.visits for child in root.children.values())
        action_probs = np.zeros(4672)

        for action, child in root.children.items():
            action_probs[action] = child.visits / total_visits

        return action_probs