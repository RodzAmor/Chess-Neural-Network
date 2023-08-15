import chess
import chess.svg
import numpy as np
from tensorflow.keras.models import load_model
from preprocessing import encode_board
"""
Uses the Minimax algorithm to evaluate the state of the board for many possibilities and finds the optimal position.

@param board:   The current board
@param depth:   How deep the algorithm will process
@param is_max:  True if the current player is max, false otherwise
@param model:   Neural network model used for evaluating the position

@return         Returns the optimal score
"""
def minimax(board, depth, is_max, model):
    pass

"""
Uses the neural network model from the parameter to evaluate the position

@param board:   The current board
@param model:   Neural network model used for evaluating the position

@return         Returns the board evalution
"""
def evaluate_board(board, model):
    encoded_board = fast_encode(board)

    # evaluate = model.predict(encoded_board)
    evaluate = model.predict(encoded_board, verbose=0)

    return evaluate[0][0]


"""
Uses the neural network model from the parameter to evaluate the position

@param board:   The current board
@param depth:   How deep the algorithm will process
@param model:   Neural network model used for evaluating the position

@return         Finds the best move for the player
"""
def find_move(board, depth, model):
    pass


piece_index = {
    'P': 0, 'R': 1, 'N': 2, 'B': 3, 'Q': 4, 'K': 5, # White Pieces
    'p': 6, 'r': 7, 'n': 8, 'b': 9, 'q': 10, 'k': 11 # Black Pieces
}

""" Slightly more optimized version of encode """
def fast_encode(board):    
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


"""
"""
def simple_move(board: chess.Board, model, is_max):
    if board.is_game_over():
        return None
    
    print(board.legal_moves)

    best_move = None
    if is_max:
        score = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            eval = evaluate_board(board, model)
            print(move, eval)
            board.pop()
            if eval > score:
                eval = score
                best_move = move
            
    else:
        score = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval = evaluate_board(board, model)
            print(move, eval)
            board.pop()
            if eval < score:
                eval = score
                best_move = move
        
    return best_move