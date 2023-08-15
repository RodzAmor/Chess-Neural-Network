import chess
import chess.svg
import numpy as np
from tensorflow.keras.models import load_model
from preprocessing import encode_board
"""
Uses the Minimax algorithm to evaluate the state of the board for many possibilities and finds the optimal position.

@param board:   The current board
@param depth:   How deep the algorithm will process. Default of 3
@param maximizing_player:  True if the current player is max, false otherwise
@param model:   Neural network model used for evaluating the position


@return         Returns the optimal score
"""
def minimax_alpha_beta(board: chess.Board, model, depth, alpha, beta, maximizing_player):
    if depth == 0 or board.is_game_over():
        return evaluate_board(board, model), None
    
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

        

"""
Uses the neural network model from the parameter to evaluate the position

@param board:   The current board
@param model:   Neural network model used for evaluating the position

@return         Returns the board evalution
"""
def evaluate_board(board, model):
    encoded_board = fast_encode(board)

    evaluate = model.predict(encoded_board)
    # evaluate = model.predict(encoded_board, verbose=0)

    return evaluate[0][0]


"""
Uses the neural network model from the parameter to evaluate the position

@param board:   The current board
@param depth:   How deep the algorithm will process
@param model:   Neural network model used for evaluating the position

@return         Finds the best move for the player
"""
def find_move(board, model, depth=3):
    maximizing_player = board.turn == chess.WHITE
    
    centipawn, best_move = minimax_alpha_beta(board, model, depth, float('-inf'), float('inf'), maximizing_player=maximizing_player)
    return centipawn, best_move


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
This function only checks with a depth of 1. It looks through every
possible next position and finds the best one.
"""
def simple_move(board: chess.Board, model, maximizing_player):
    if board.is_game_overame_over():
        return None
    
    # print(board.legal_moves)

    best_move = None
    if maximizing_player:
        score = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            eval = evaluate_board(board, model)
            # print(move, eval)
            board.pop()
            if eval > score:
                eval = score
                best_move = move
            
    else:
        score = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval = evaluate_board(board, model)
            # print(move, eval)
            board.pop()
            if eval < score:
                eval = score
                best_move = move
        
    return best_move