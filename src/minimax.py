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
    pass

"""
Uses the neural network model from the parameter to evaluate the position

@param board:   The current board
@param depth:   How deep the algorithm will process
@param model:   Neural network model used for evaluating the position

@return         Finds the best move for the player
"""
def find_move(board, depth, model):
    pass

def play_game():
    pass