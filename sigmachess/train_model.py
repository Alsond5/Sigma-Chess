# train_model.py

import chess.engine
from classes import GameState, MCTS, ReplayBuffer
from models import sigmachess_network, create_model

import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
import os

# Harici Chess Engine Yolu
engine = chess.engine.SimpleEngine.popen_uci(r"/home/alsond5/projects/sigmachess/stockfish/stockfish-ubuntu-x86-64-avx2")

def self_play(mcts, num_games=100):
    replay_buffer = ReplayBuffer(maxlen=500000)
    
    for game in range(num_games):
        state = GameState()
        temperature = 1.0 if game < 30 else 0.1

        states, policies, rewards = [], [], []
        player = np.random.choice([chess.WHITE, chess.BLACK])

        while not state.is_terminal():
            if state.board.turn == player:
                action_probs = mcts.run(state, temperature)
                
                states.append(state.get_current_state())
                policies.append(action_probs)

                action = np.random.choice(len(action_probs), p=action_probs)
                state.get_next_state(action)
            else:
                result = engine.play(state.board, chess.engine.Limit(0.5))
                state.apply_action(result.move)

        winner = state.get_winner()
        value = 1 if winner == player else (0 if winner == 2 else -1)

        print(player, state.board.board_fen())

        for s, p, v in zip(states, policies, [value]):
            replay_buffer.store(s, p, v)

    return replay_buffer

def create_callbacks(checkpoint_path="checkpoints/model_checkpoint.weights.h5"):
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

def train_model(model, replay_buffer: ReplayBuffer, batch_size=256, epochs=3, checkpoint_path="checkpoints/model_checkpoint.weights.h5"):
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
            # callbacks=callbacks,
            verbose=1
        )

def train_sigmachess(model, num_iterations=100, num_games_per_iteration=100):
    for iteration in range(num_iterations):
        print(f"Iteration {iteration + 1}/{num_iterations}")
        
        mcts = MCTS(model, 1.0, 80)
        replay_buffer = self_play(mcts, num_games_per_iteration)
        train_model(model, replay_buffer)

    model.save("sigmachess_model/full_model.keras")

model = create_model()
train_sigmachess(model, num_iterations=7000, num_games_per_iteration=15)