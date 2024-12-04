import numpy as np # linear algebra
import math

import chess

from collections import deque

import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import HeNormal

import chess.engine
from tensorflow.keras.callbacks import ModelCheckpoint

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

class GameState:
    row = 8
    col = 8
    promotion_indexes = {
        chess.KNIGHT: 0,
        chess.ROOK: 1,
        chess.BISHOP: 2
    }
    
    def __init__(self, player_color: chess.Color) -> None:
        self.board = chess.Board()
        self.repetition_count = 0
        self.player_color: chess.Color = player_color

    def get_initial_state(self):
        self.board.reset()

        return self.get_current_state()
        
    def get_current_state(self, T=8):
        input_tensor = np.zeros((8, 8, 119), dtype=np.float32)

        for t in range(T):
            _t = T - t - 1
            if len(self.board.move_stack) < _t:
                continue
            
            self.create_input(input_tensor, _t)

        color = 0 if self.board.turn == chess.WHITE else 1
        input_tensor[:, :, 112] = color

        input_tensor[:, :, 113] = self.board.fullmove_number / 500

        p1_castling = (1 * self.board.has_kingside_castling_rights(chess.WHITE)) | (2 * self.board.has_queenside_castling_rights(chess.WHITE))
        p1_castling_bit = format(p1_castling, "02b")
        input_tensor[:, :, 114] = int(p1_castling_bit[0])
        input_tensor[:, :, 115] = int(p1_castling_bit[1])

        p2_castling = (1 * self.board.has_kingside_castling_rights(chess.BLACK)) | (2 * self.board.has_queenside_castling_rights(chess.BLACK))
        p2_castling_bit = format(p2_castling, "02b")
        input_tensor[:, :, 116] = int(p2_castling_bit[0])
        input_tensor[:, :, 117] = int(p2_castling_bit[1])

        input_tensor[:, :, 118] = self.board.halfmove_clock / 50

        return np.expand_dims(input_tensor, axis=0)

    def get_next_state(self, action: int):
        source_index = action // 73
        destination_index = 0
        move_type = action % 73
        
        promotion = None

        if move_type < 56:
            direction = move_type // 7
            movement = (move_type % 7) + 1

            destination_index = source_index + (movement * 8) if direction == 0 else destination_index
            destination_index = source_index + (movement * 9) if direction == 1 else destination_index
            destination_index = source_index + movement if direction == 2 else destination_index
            destination_index = source_index + (movement * -7) if direction == 3 else destination_index
            destination_index = source_index + (movement * -8) if direction == 4 else destination_index
            destination_index = source_index + (movement * -9) if direction == 5 else destination_index
            destination_index = source_index + (-movement) if direction == 6 else destination_index
            destination_index = source_index + (movement * 7) if direction == 7 else destination_index
        elif move_type >= 56 and move_type < 64:
            direction = move_type - 56

            destination_index = source_index + 17 if direction == 0 else destination_index
            destination_index = source_index + 10 if direction == 1 else destination_index
            destination_index = source_index - 6 if direction == 2 else destination_index
            destination_index = source_index - 15 if direction == 3 else destination_index
            destination_index = source_index - 17 if direction == 4 else destination_index
            destination_index = source_index - 10 if direction == 5 else destination_index
            destination_index = source_index + 6 if direction == 6 else destination_index
            destination_index = source_index + 15 if direction == 7 else destination_index
        else:
            direction = (move_type - 64) // 3
            promotion_index = (move_type - 64) % 3

            promotion = chess.KNIGHT if promotion_index == 0 else (chess.ROOK if promotion_index == 1 else chess.BISHOP)

            color_value = 1 if self.board.turn == chess.WHITE else -1

            if direction == 0:
                destination_index = source_index + (8 * color_value)
            elif direction == 1:
                destination_index = source_index + (9 * color_value)
            else:
                destination_index = source_index + (7 * color_value)

        from_square = chess.Square(source_index)
        to_square = chess.Square(destination_index)

        promotion_rank = 7 if self.board.turn == chess.WHITE else 0

        if promotion is None:
            if self.board.piece_type_at(from_square) == chess.PAWN and chess.square_rank(to_square) == promotion_rank:
                promotion = chess.QUEEN
        
        move = chess.Move(from_square, to_square, promotion)

        self.apply_action(move)

        return move, self.get_current_state()
    
    def apply_action(self, move: chess.Move):
        try:
            self.board.push(move)
        except Exception as e:
            print(list(self.board.legal_moves))
            print(self.get_valid_moves())

            print(e)

            raise Exception("Error")
    
    def create_input(self, input_tensor: np.ndarray, t: int):
        piece_types = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5
        }

        board = self.board.copy()
        for _ in range(t):
            board.pop()

        transposition_key = board._transposition_key()

        for square in chess.SQUARES:
            piece = board.piece_at(square)

            if piece is None:
                continue
            
            piece_index = piece_types[piece.piece_type]
            piece_color = 0 if piece.color == chess.WHITE else 1

            index = (t * 14) + (piece_color * 6) + piece_index
            input_tensor[square // 8][square % 8][index] = 1

        repetition_count = 0
        index = (t * 14) + 12
        
        try:
            while board.move_stack:
                move = board.pop()
                if board.is_irreversible(move):
                    break

                if board._transposition_key() == transposition_key:
                    repetition_count += 1

                if repetition_count == 3:
                    break
        finally:
            repetition_count = 3 if repetition_count > 3 else repetition_count

            repetition_count_bits = [int(x) for x in format(repetition_count, "02b")]
            input_tensor[:, :, index] = repetition_count_bits[0]
            input_tensor[:, :, index + 1] = repetition_count_bits[1]
            
    def get_valid_moves(self):
        legal_moves = []

        for valid_move in self.board.legal_moves:
            s_row, s_col, from_square_index = self.index_of_square(valid_move.from_square)
            d_row, d_col, to_square_index = self.index_of_square(valid_move.to_square)
            
            if valid_move.promotion:
                direction = self.direction_of_move_for_ray_directions(s_row, s_col, d_row, d_col)

                if valid_move.promotion == chess.QUEEN:                    
                    index = (from_square_index * 73) + (direction * 7)
                    legal_moves.append(index)
                else:
                    promotion_index = self.promotion_indexes[valid_move.promotion]

                    if direction > 2 and direction < 6:
                        direction = 0 if direction == 4 else (1 if direction == 5 else 2)
                    elif direction == 7:
                        direction = 2

                    index = (from_square_index * 73) + ((direction * 3) + promotion_index + 64)
                    legal_moves.append(index)
            elif self.board.piece_type_at(valid_move.from_square) == chess.KNIGHT:
                direction = self.direction_of_move_for_knights(s_row, s_col, d_row, d_col)
                
                index = (from_square_index * 73) + direction + 56
                legal_moves.append(index)

            else:
                direction = self.direction_of_move_for_ray_directions(s_row, s_col, d_row, d_col)
                count_of_square = self.count_of_square_for_movement(s_row, s_col, d_row, d_col) - 1

                index = (from_square_index * 73) + ((direction * 7) + count_of_square)
                legal_moves.append(index)

        return legal_moves

    def index_of_square(self, square: chess.Square):
        row = chess.square_rank(square)
        col = chess.square_file(square)
        index = (row * 8) + col

        return row, col, index

    def direction_of_move_for_ray_directions(self, s_row: int, s_col: int, d_row: int, d_col: int):
        delta_x = d_col - s_col
        delta_y = d_row - s_row

        if delta_x == 0:
            return 0 if delta_y > 0 else 4
        
        if delta_y == 0:
            return 2 if delta_x > 0 else 6

        if delta_x < 0:
            return 7 if delta_y > 0 else 5

        return 1 if delta_y > 0 else 3
    
    def direction_of_move_for_knights(self, s_row: int, s_col: int, d_row: int, d_col: int):
        delta_x = d_col - s_col
        delta_y = d_row - s_row

        if delta_x == 1:
            return 0 if delta_y > 0 else 3
        
        if delta_x == 2:
            return 1 if delta_y > 0 else 2

        if delta_x == -1:
            return 7 if delta_y > 0 else 4

        return 6 if delta_y > 0 else 5

    def count_of_square_for_movement(self, s_row: int, s_col: int, d_row: int, d_col: int):
        delta_x = d_col - s_col
        delta_y = d_row - s_row

        return max(abs(delta_x), abs(delta_y))
    
    def get_winner(self):
        result = self.board.result()

        if result == "1-0":
            return chess.WHITE
        
        if result == "0-1":
            return chess.BLACK
        
        return 2
    
    def is_terminal(self):
        return self.board.is_game_over()
    
    def clone(self):
        cloned_state = GameState(self.player_color)
        cloned_state.board = self.board.copy()

        return cloned_state

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

        if len(self.children.items()) > 0:
            self.is_expanded = True

    def select(self, c_puct=1.0):
        max_ucb = -float('inf')
        best_action = None
        best_child = None
        
        for action, child in self.children.items():
            ucb = self.calculate_ucb(child, c_puct)
            
            if ucb > max_ucb:
                max_ucb = ucb
                best_action = action
                best_child = child

        if best_child is None:
            print(self.visits, ucb, child.value)
            for action, child in self.children.items():
                print(action, child.visits, child.value_sum, child.prior_prob)
                
        return best_action, best_child

    def calculate_ucb(self,child, c_puct):
        ucb = c_puct * child.prior_prob * (math.sqrt(self.visits) / (1 + child.visits))
        value = -child.value if child.visits > 0 else 0

        return ucb + value

    def backup(self, value):
        if value == float("nan"):
            print("Burası çok önemli:", value)
            
        self.visits += 1
        self.value_sum += value
        if self.parent:
            self.parent.backup(-value)

class MCTS:
    def __init__(self, model, c_puct=1.0, simulations=50):
        self.model = model
        self.c_puct = c_puct
        self.simulations = simulations

    def run(self, initial_state, temperature=1.0):
        root = Node(initial_state)
        
        # First evaluate and expand root
        action_probs, value = self.evaluate(initial_state)
        valid_moves = initial_state.get_valid_moves()
        
        # Add Dirichlet noise to root (alpha=0.3 for chess)
        noise = np.random.dirichlet([0.3] * len(valid_moves))
        action_probs[valid_moves] = action_probs[valid_moves] * 0.75 + noise[[x for x in range(len(valid_moves))]] * 0.25

        root.expand(action_probs)

        for _ in range(self.simulations):
            node = root
            
            # Selection
            while node.is_expanded and not node.state.is_terminal():
                action, node = node.select(self.c_puct)
            
            # Expansion and Evaluation
            if not node.state.is_terminal():
                action_probs, value = self.evaluate(node.state)
                node.expand(action_probs)
            else:
                winner = node.state.get_winner()
                value = 1 if winner == node.state.player_color else (0 if winner == 2 else -1)
            
            # Backup
            node.backup(value)

        return self.get_action_probs(root, temperature)

    def evaluate(self, state):
        state_tensor = state.get_current_state()
        # state_tensor = np.expand_dims(state_tensor, axis=0)
        
        policy, value = self.model.predict(state_tensor, verbose=0)
        policy = policy[0]
        
        # Mask invalid moves
        valid_moves = state.get_valid_moves()
        mask = np.zeros(policy.shape)
        mask[valid_moves] = 1
        
        policy *= mask
        
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

class ReplayBuffer:
    def __init__(self, maxlen=500000):
        self.buffer = deque(maxlen=maxlen)
        self.current_size = 0
        self.lock = threading.Lock()

    def store(self, state, policy, value):
        """Store a single game state transition"""
        self.buffer.append({
            'state': state,
            'policy': policy,
            'value': value
        })
        self.current_size = len(self.buffer)

    def store_multiple_data(self, states, policies, value):
        with self.lock:
            for s, p, v in zip(states, policies, [value]):
                self.store(s, p, v)

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

# tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

# tf.tpu.experimental.initialize_tpu_system(tpu)
# tpu_strategy = tf.distribute.TPUStrategy(tpu)

print("All devices: ", tf.config.list_logical_devices('GPU'))

policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

def residual_block(inputs, filters=256, kernel_size=(3, 3), stride=1):
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, kernel_size, strides=stride, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([x, inputs])
    x = layers.ReLU()(x)
    
    return x

def sigmachess_network(input_shape=(8, 8, 119)):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(256, (3, 3), strides=1, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    for _ in range(19):
        x = residual_block(x)

    policy = layers.Conv2D(256, (3, 3), strides=1, padding="same")(x)
    policy = layers.BatchNormalization()(policy)
    policy = layers.ReLU()(policy)
    policy = layers.Conv2D(73, (1, 1), strides=1, padding="same")(policy)
    policy = layers.Flatten()(policy)
    policy = layers.Dense(4672, name="policy_output", activation="softmax")(policy)

    value = layers.Conv2D(1, (1, 1), strides=1, padding="same")(x)
    value = layers.BatchNormalization()(value)
    value = layers.ReLU()(value)
    value = layers.Flatten()(value)
    value = layers.Dense(256, activation="relu")(value)
    value = layers.Dense(1, activation="tanh", name="value_output")(value)

    model = models.Model(inputs=inputs, outputs=[policy, value])

    return model

def create_model():
    model = sigmachess_network()

    model.compile(
        optimizer=Adam(learning_rate=0.02),
        loss={
            "policy_output": "categorical_crossentropy",
            "value_output": "mean_squared_error"
        },
        metrics={
            "policy_output": "accuracy",
            "value_output": "mse"
        }
    )
    
    return model

# train_model.py

def play_vs_stockfish(model, game, replay_buffer):
    temperature = 1.0 if game < 5 else 0.1

    w_states, w_policies, w_rewards = [], [], []
    player = np.random.choice([chess.WHITE, chess.BLACK])

    state = GameState(player)

    engine = chess.engine.SimpleEngine.popen_uci(r"/kaggle/working/stockfish-ubuntu-x86-64-avx2")

    while not state.is_terminal():
        if state.board.turn == player:
            mcts = MCTS(model, 1.0, 10)
            action_probs = mcts.run(state, temperature)
            
            w_states.append(state.get_current_state())
            w_policies.append(action_probs)
    
            action = np.random.choice(len(action_probs), p=action_probs)
            state.get_next_state(action)
        else:
            result = engine.play(state.board, chess.engine.Limit(0.04))
            state.apply_action(result.move)

    engine.close()

    winner = state.get_winner()
    w_value = 1 if winner == player else (0 if winner == 2 else -1)

    print(player, state.board.board_fen())

    replay_buffer.store_multiple_data(w_states, w_policies, w_value)

def self_play(model, num_games=100, max_workers=5):
    replay_buffer = ReplayBuffer(maxlen=500000)

    for i in range(num_games):
        play_vs_stockfish(model, i, replay_buffer)

    return replay_buffer

def create_callbacks(checkpoint_path="/kaggle/working/sigma_checkpoint.weights.h5"):
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor="loss",
        mode="min",
        save_best_only=True,
        save_freq="epoch",
        verbose=1
    )
    
    return [checkpoint]

def train_model(model, replay_buffer: ReplayBuffer, batch_size=64, epochs=1, checkpoint_path="/kaggle/working/sigma_checkpoint.weights.h5"):
    callbacks = create_callbacks(checkpoint_path)

    for epoch in range(epochs):
        # replay_buffer.augment_data()
        states, policies, values = replay_buffer.sample(batch_size)

        states = np.squeeze(states)
        if len(states.shape) == 3:  # Eğer eksik boyut varsa
            states = np.expand_dims(states, -1)

        values = np.array(values).reshape(-1, 1)
        
        model.fit(
            states,
            { "policy_output": policies, "value_output": values },
            batch_size=batch_size,
            epochs=1,
            callbacks=callbacks,
            verbose=1
        )

is_stop = False

def train_sigmachess(model, num_iterations=100, num_games_per_iteration=100):
    global is_stop
    
    for iteration in range(num_iterations):
        print(f"Iteration {iteration + 1}/{num_iterations}")
        
        replay_buffer = self_play(model, num_games_per_iteration)
        train_model(model, replay_buffer)

        print()

        if is_stop:
            break

    model.save("/kaggle/working/sigma_model.keras")

# with tpu_strategy.scope():
    # model = create_model()

def stop():
    global is_stop
    
    while True:
        inp = input("")
        if inp == "stop":
            is_stop = True
            print("After the iteration is completed, the training will be stopped and the model will be saved!")
            
            break

t = threading.Thread(target=stop, daemon=True)
t.start()

model = create_model()
train_sigmachess(model, num_iterations=700, num_games_per_iteration=10)