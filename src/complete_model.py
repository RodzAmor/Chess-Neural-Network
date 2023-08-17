import chess
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

class ChessModel:
    def __init__(self, model_path=None):
        if model_path == None:
            self.nn_model = self.build_model()
        else:
            self.nn_model = load_model(model_path)

    def build_model(self):
        nn_model = Sequential([
            Conv2D(32, (3, 3), activation="relu", input_shape=(8, 8, 13)),
            MaxPooling2D((2, 2)),
            Conv2D(32, (3, 3), activation="relu"),
            Flatten(),
            Dense(64, activation="relu"),
            Dense(1)
        ])

        print(nn_model.summary())
        
        nn_model.compile(optimizer='adam', loss='mean_squared_error')
        # model.compile(optimizer='adam', loss='mean_squared_error', metrics=[])
        
        return nn_model

    def train_model(self, X_train, y_train, epochs=100, batch_size=32):
        self.nn_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    
        # return trained_model
    
    def evaluate_model(self, model, X_test, y_test):
        pass

    def find_move(self, board, model, depth=3):
        maximizing_player = board.turn == chess.WHITE
        
        centipawn, best_move = minimax_alpha_beta(board, model, depth, float('-inf'), float('inf'), maximizing_player=maximizing_player)
        return centipawn, best_move
    
def minimax_alpha_beta(board: chess.Board, model, depth, alpha, beta, maximizing_player):
    if depth == 0 or board.is_game_over():
        return evaluate_board(fast_encode(board)), None
    
    legal_moves = list(board.legal_moves)
    best_move = None

    if maximizing_player:
        maxEval = float('-inf')

        for move in legal_moves:
            board.push(move)
            eval, _ = minimax_alpha_beta(board, model, depth - 1, alpha, beta, False)
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
            eval, _ = minimax_alpha_beta(board, model, depth - 1, alpha, beta, True)
            board.pop()
            # minEval = min(minEval, eval)

            if eval < minEval:
                minEval = eval
                best_move = move

            beta = min(beta, eval)
            if beta <= alpha:
                break

        return minEval, best_move
        
# @tf.function
def evaluate_board(encoded_board, model):
    evaluate = model.nn_model.predict(encoded_board)
    # evaluate = model.predict(encoded_board, verbose=0)

    return evaluate[0][0]
    
def fast_encode(board):
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
    

def load_data(board_path, evals_path):
    boards = np.load(board_path, allow_pickle=True)
    evals = np.load(evals_path, allow_pickle=True)

    return boards, evals


def split_data(boards, evals, test_size=0.15):
    X_train, X_test, y_train, y_test = train_test_split(boards, evals, test_size=test_size)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    chess_model = ChessModel()
    
    board_path = "data/processed/sample_boards.npy"
    evals_path = "data/processed/sample_evals.npy"

    boards, evals = load_data(board_path, evals_path)

    X_train, X_test, y_train, y_test = split_data(boards, evals)
    
    chess_model.train_model(X_train, y_train)
    chess_model.nn_model.save('test_chess_model')
