import chess
import numpy as np

class GameState:
    row = 8
    col = 8
    promotion_indexes = {
        chess.KNIGHT: 0,
        chess.ROOK: 1,
        chess.BISHOP: 2
    }
    
    def __init__(self) -> None:
        self.board = chess.Board()
        self.repetition_count = 0
        self.player_color: chess.Color = chess.WHITE

    def get_initial_state(self):
        self.board.reset()

        return self.get_current_state()
        
    def get_current_state(self, T=8):
        input_tensor = np.zeros((8, 8, 119), dtype=np.uint8)

        for t in range(T):
            _t = T - t - 1
            if len(self.board.move_stack) < _t:
                continue
            
            self.create_input(input_tensor, _t)

        color = 0 if self.board.turn == chess.WHITE else 1
        input_tensor[:, :, 112] = color

        input_tensor[:, :, 113] = len(self.board.move_stack) > 0

        p1_castling = (1 * self.board.has_kingside_castling_rights(chess.WHITE)) | (2 * self.board.has_queenside_castling_rights(chess.WHITE))
        p1_castling_bit = format(p1_castling, "02b")
        input_tensor[:, :, 114] = int(p1_castling_bit[0])
        input_tensor[:, :, 115] = int(p1_castling_bit[1])

        p2_castling = (1 * self.board.has_kingside_castling_rights(chess.BLACK)) | (2 * self.board.has_queenside_castling_rights(chess.BLACK))
        p2_castling_bit = format(p2_castling, "02b")
        input_tensor[:, :, 116] = int(p2_castling_bit[0])
        input_tensor[:, :, 117] = int(p2_castling_bit[1])

        input_tensor[:, :, 118] = int(self.board.is_fifty_moves())

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
            direction = move_type // 3
            promotion_index = move_type % 3

            promotion = chess.KNIGHT if promotion_index == 0 else (chess.ROOK if promotion_index == 1 else chess.BISHOP)

            if direction == 0:
                destination_index = source_index + (8 * (self.board.turn != chess.WHITE) * -1)
            elif direction == 1:
                destination_index = source_index + (9 * (self.board.turn != chess.WHITE) * -1)
            else:
                destination_index = source_index + (7 * (self.board.turn != chess.WHITE) * -1)

        from_square = chess.Square(source_index)
        to_square = chess.Square(destination_index)

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

                    if direction > 2:
                        direction = 0 if direction == 4 else (1 if direction == 5 else 2)
                    else:
                        direction = 2 if direction == 7 else direction

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
        cloned_state = GameState()
        cloned_state.board = self.board.copy()

        return cloned_state