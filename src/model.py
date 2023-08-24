import chess
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

class MinimalChessModel:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        # self.transposition_table = {}

    def find_move(self, board, depth=3):
            maximizing_player = board.turn == chess.WHITE
            
            centipawn, best_move = self.minimax_alpha_beta(board, depth, float('-inf'), float('inf'), maximizing_player=maximizing_player)
            print(centipawn, best_move)
            return centipawn, best_move    

    def minimax_alpha_beta(self, board: chess.Board, depth, alpha, beta, maximizing_player):
        if depth == 0 or board.is_game_over():
            return self.evaluate_board2(self.fast_encode(board)), None
        
        # position_key = board.board_fen()
        # if position_key in self.transposition_table:
        #     return self.transposition_table[position_key]
        

        legal_moves = list(board.legal_moves)
        legal_moves.sort(key=lambda move: -board.is_capture(move)) # Move ordering
        best_move = None

        if maximizing_player:
            maxEval = float('-inf')

            for move in legal_moves:
                board.push(move)
                eval, _ = self.minimax_alpha_beta(board, depth - 1, alpha, beta, False)
                board.pop()
                # maxEval = max(maxEval, eval)

                if eval > maxEval:
                    maxEval = eval
                    best_move = move

                alpha = max(alpha, eval)

                if beta <= alpha:
                    break
            
            # self.transposition_table[position_key] = (maxEval, best_move)
            return maxEval, best_move
        else:
            minEval = float('inf')

            for move in legal_moves:
                board.push(move)
                eval, _ = self.minimax_alpha_beta(board, depth - 1, alpha, beta, True)
                board.pop()
                # minEval = min(minEval, eval)

                if eval < minEval:
                    minEval = eval
                    best_move = move

                beta = min(beta, eval)
                if beta <= alpha:
                    break

            # self.transposition_table[position_key] = (minEval, best_move)
            return minEval, best_move

    @tf.function
    def evaluate_board2(self, encoded_board):
        return self.model(encoded_board, training=False)

    def evaluate_board(self, encoded_board):
        return self.model.predict(encoded_board)

    # Function to encode a chess board to a tensor
    def fast_encode(self, board):
        piece_index = {
            'P': 0, 'R': 1, 'N': 2, 'B': 3, 'Q': 4, 'K': 5,  # White Pieces
            'p': 0, 'r': 1, 'n': 2, 'b': 3, 'q': 4, 'k': 5  # Black Pieces (will use negative values)
        }

        encoded_board = np.zeros((8, 8, 6))  # Initialize with zeros

        # Fill the values of each tensor
        for i in range(8):
            for j in range(8):
                piece = board.piece_at(chess.square(j, 7 - i))
                if piece:
                    # If it's a white piece, set value to 1. If black, set to -1.
                    value = 1 if piece.symbol().isupper() else -1
                    encoded_board[i, j, piece_index[piece.symbol()]] = value

        return encoded_board.reshape(1, 8, 8, 6)
    