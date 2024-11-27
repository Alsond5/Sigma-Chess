# train_model.py

import chess.engine
from classes import GameState, MCTS, ReplayBuffer
from models import sigmachess_network

import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
import os

# Harici Chess Engine Yolu
engine = chess.engine.SimpleEngine.popen_uci(r"/home/alsond5/projects/sigmachess/stockfish/stockfish-ubuntu-x86-64-avx2")

def self_play(mcts, num_games=100):
    replay_buffer = ReplayBuffer()
    
    for _ in range(num_games):
        state = GameState()
        states, policies, rewards = [], [], []
        player = np.random.choice([chess.WHITE, chess.BLACK])

        while not state.is_terminal():
            if state.board.turn == player:
                action_probs = mcts.run(state)
                action = np.random.choice(len(action_probs), p=action_probs)
                
                s = state.get_current_state()
                states.append(s)
                policies.append(action_probs)

                state.get_next_state(action)
            else:
                result = engine.play(state.board, chess.engine.Limit(0.5))
                state.apply_action(result.move)

        winner = state.get_winner()
        reward = 1 if winner == player else (0 if winner == 2 else -1)

        rewards = [reward] * len(states)

        print(reward)

        for s, p, r in zip(states, policies, rewards):
            replay_buffer.store(s, p, r)

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

def train_model(model, replay_buffer, batch_size=128, epochs=20, checkpoint_path="checkpoints/model_checkpoint.weights.h5"):
    callbacks = create_callbacks(checkpoint_path)

    for epoch in range(epochs):
        states, policies, values = replay_buffer.sample(batch_size)
        
        # Boyutları doğrulamak için sıkıştırma ve yeniden şekillendirme
        states = np.squeeze(states)
        if len(states.shape) == 3:  # Eğer eksik boyut varsa
            states = np.expand_dims(states, -1)
        
        model.fit(
            { "input_layer": states },
            { "policy_output": policies, "value_output": values },
            batch_size=batch_size,
            epochs=1,
            callbacks=callbacks,
            verbose=1
        )

def train_sigmachess(model, num_iterations=100, num_games_per_iteration=100):
    for iteration in range(num_iterations):
        print(f"Iteration {iteration + 1}/{num_iterations}")
        
        mcts = MCTS(model)  # Aynı MCTS'i yeniden kullan
        replay_buffer = self_play(mcts, num_games_per_iteration)
        train_model(model, replay_buffer)

    model.save("sigmachess_model/full_model.keras")

model = sigmachess_network()
if os.path.exists("checkpoints/model_checkpoint.weights.h5"):
    model.load_weights("checkpoints/model_checkpoint.weights.h5")

train_sigmachess(model, num_iterations=10, num_games_per_iteration=20)