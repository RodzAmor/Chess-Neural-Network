import chess
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

class ChessModel:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def find_move(self, board, depth=3):
            maximizing_player = board.turn == chess.WHITE
            
            centipawn, best_move = self.minimax_alpha_beta(board, depth, float('-inf'), float('inf'), maximizing_player=maximizing_player)
            print(best_move)
            return centipawn, best_move    

    def minimax_alpha_beta(self, board: chess.Board, depth, alpha, beta, maximizing_player):
        if depth == 0 or board.is_game_over():
            return self.evaluate_board(self.fast_encode(board)), None
        
        legal_moves = list(board.legal_moves)
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

            return minEval, best_move

    @tf.function
    def evaluate_board(self, encoded_board):
        return self.model(encoded_board, training=False)

        
    def fast_encode(self, board):
        piece_index = {
            'P': 0, 'R': 1, 'N': 2, 'B': 3, 'Q': 4, 'K': 5, # White Pieces
            'p': 6, 'r': 7, 'n': 8, 'b': 9, 'q': 10, 'k': 11 # Black Pieces
        }

        encoded_board = np.zeros((8, 8, 13), dtype=np.float32)

        # Fill the values of each tensor
        for i in range(8):
            for j in range(8):
                piece = board.piece_at(chess.square(j, 7 - i))

                if piece != None:
                    encoded_board[i, j, piece_index[piece.symbol()]] = 1
                else:
                    encoded_board[i, j, 12] = 1

        return encoded_board.reshape((1, 8, 8, 13))