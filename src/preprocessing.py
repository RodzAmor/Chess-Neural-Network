import chess.pgn
import numpy as np
import pandas as pd

"""
Reads a PGN file and extracts each game
Default is to process every game. Set the value of num_games in order to 

@param filepath: String of the file to process
@param num_games: (Optional) number of games to process
@return list of games    
"""
# @returns  list of games
def process_pgn(file_path, num_games=float('inf')):
    games = []

    with open(file_path) as pgn:
        for _ in range(num_games):
            game = chess.pgn.read_game(pgn)

            # All games are processed
            if game is None: 
                break

            games.append(game)

    return games

"""
Encodes the board into an 8x8 shape so that it can be processed
Need to convert the board into a format that can be used in a neural network
"""
def encode_board(board):
    pass

"""
Return the labels from the game such as the eval or outcome (win, loss, draw, stalemate)
"""
def extract_game_labels(game):
    pass
"""
Extracts a list of games and creates a pandas dataframe
Saves it into a csv for further processing

Example: preprocess_games('data/raw/sample.pgn', 'data/raw/sample.csv')
"""
def preprocess_games(file_path, save_path, num_games=float('inf')):
    games = process_pgn(file_path, num_games)

    data = []

    for game in games:
        board = game.board()

        for move in game.mainline_moves():
            pass






