from classes import GameState, Node, MCTS
from models import sigmachess_network

import numpy as np
import chess

import socket

model = sigmachess_network()
model.load_weights("/home/alsond5/projects/sigmachess/checkpoints/model_checkpoint.weights.h5")

gamestate = GameState()
mcts = MCTS(model, simulations=10)

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("172.29.229.122", 5051))

def start_server():
    s.listen()

    client_socket, client_address = s.accept()

    while client_socket:
        data = client_socket.recv(1024)
        move = data.decode()

        print(move)

        if move == "exit":
            s.close()
            break
        
        gamestate.apply_action(chess.Move.from_uci(move))

        best_move = mcts.run(gamestate)
        action = np.random.choice(len(best_move), p=best_move)
        m, state = gamestate.get_next_state(action)

        client_socket.sendall(chess.Move.uci(m).encode())

try:
    start_server()
except KeyboardInterrupt:
    s.close()

finally:
    s.close()