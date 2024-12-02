# train_model.py

import chess.engine
from classes import GameState, MCTS, ReplayBuffer
from models import sigmachess_network, create_model

import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
import os

# Harici Chess Engine Yolu
engine = chess.engine.SimpleEngine.popen_uci(r"/home/alsond5/projects/sigmachess/stockfish/stockfish-ubuntu-x86-64-avx2")

def augment_data(replay_buffer):
    augmented_buffer = []
    for sample in replay_buffer:
        state = sample["state"]
        policy = sample["policy"]
        value = sample["value"]

        # Örneğin 90 derece döndürme
        rotated_state = np.rot90(state)
        rotated_policy = np.rot90(policy)  # Politika dönüşümü

        # Aynalama
        flipped_state = np.flip(state, axis=1)
        flipped_policy = np.flip(policy, axis=1)

        # Orijinal ve artırılmış örnekleri ekle
        augmented_buffer.append({"state": state, "policy": policy, "value": value})
        augmented_buffer.append({"state": rotated_state, "policy": rotated_policy, "value": value})
        augmented_buffer.append({"state": flipped_state, "policy": flipped_policy, "value": value})

    return augmented_buffer

def self_play(mcts, num_games=100):
    replay_buffer = ReplayBuffer()
    
    for _ in range(num_games):
        state = GameState()
        states, policies, rewards = [], [], []
        player = np.random.choice([chess.WHITE, chess.BLACK])

        while not state.is_terminal():
            if state.board.turn == player:
                action_probs = mcts.run(state, 1.0)
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

        print(rewards)

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

def train_model(model, replay_buffer: ReplayBuffer, batch_size=128, epochs=20, checkpoint_path="checkpoints/model_checkpoint.weights.h5"):
    callbacks = create_callbacks(checkpoint_path)

    for epoch in range(epochs):
        replay_buffer.augment_data()
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
        
        mcts = MCTS(model)  # Aynı MCTS'i yeniden kullan
        replay_buffer = self_play(mcts, num_games_per_iteration)
        train_model(model, replay_buffer)

    model.save("sigmachess_model/full_model.keras")

model = create_model()
train_sigmachess(model, num_iterations=10, num_games_per_iteration=1)