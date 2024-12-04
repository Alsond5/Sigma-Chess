# train_model.py

import chess.engine
from classes import GameState, MCTS, ReplayBuffer
from models import sigmachess_network, create_model

import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
import os

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