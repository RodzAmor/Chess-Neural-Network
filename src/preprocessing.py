# from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import chess.pgn
import chess.engine
import numpy as np
import pandas as pd
import re
import logging
from tqdm import tqdm # For progress bar
import threading
import os # Used for tracking the # of processes in multi-process execution

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
games_evaluated = 0


import sys
sys.setrecursionlimit(5000) # or a higher value


"""
Reads a PGN file and extracts each game
Default is to process every game. Set the value of num_games in order to 

@param filepath: String of the file to process
@param num_games: (Optional) number of games to process
@return list of games    
"""
# @returns  list of games
def process_pgn(file_path, num_games=float('inf')):
    logging.info(f"Processing PGN at {file_path}")
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

Board is represented with a shape of (8,8,13) for an 8x8 grid with 13 possible states including the empty state

@return numpy dataframe with shape (8, 8, 13)
"""
def encode_board(board: chess.Board):
    # P - Pawn, R - Rook, B - Bishop, N - Knight, Q - Queen, K - King
    # 6 black pieces + 6 white pieces + 1 empty slot = 13 possible tile states
    # As a result, the shape of each board state is 8x8x13 for the 8x8 grid and 13 possible states
    white_pieces = {piece: index for index, piece in enumerate("PRNBQK")}
    black_pieces = {piece: index + 5 for index, piece in enumerate("prnbqk")}
    all_pieces = white_pieces | black_pieces # Concatenate the two dictionaries

    encoded_board = np.zeros((8, 8, 13))
    # Fill the values of each tensor
    for i in range(8):
        for j in range(8):
            piece = board.piece_at(chess.square(j, 7 - i))

            if piece != None:
                encoded_board[i, j, all_pieces[piece.symbol()]] = 1
            else:
                encoded_board[i, j, 12] = 1

    return encoded_board
    

"""
Return the evaluation score to be used as labels
The evaluation is provided in the pgn from the lichess database which uses stockfish.
"""
# def extract_eval(board_state, label):
#     if "#" in label:
#         if label[1] == "-": # Checkmate for black
#             return -1000
#         else:
#             return 1000

#     eval = re.search(r'\[%eval (.*?)\]', label)

#     if eval != None:
#         return float(eval.group(1))
#     else:
#         return get_stockfish_eval(board_state)


"""
Uses the stockfish engine as of August 2023 for games that do not have the stockfish evaluation included already into the pgn.

@param the board state
@return float of stockfish evaluation score from white's perspective
"""
def get_stockfish_eval(boards):
    stockfish = "/opt/homebrew/bin/stockfish"
    # stockfish = "data\stockfish\stockfish-windows-x86-64-avx2.exe"
    eval_time = 0.1 # Feel free to modify for longer evaluation of positions
    scores = []


    with chess.engine.SimpleEngine.popen_uci(stockfish) as engine:
        for board in boards:
            eval = engine.analyse(board, chess.engine.Limit(time=eval_time))
            global games_evaluated
            games_evaluated += 1

            score = eval['score'].white().score(mate_score=1000) / 100.0 # centipawn
            # print(board, score)
            # print(board, score)
            # logging.info(f"Performing Stockfish evaluation: {games_evaluated} board states evaluated. Eval: {score}")
        
            scores.append(score)

    return scores
    

"""
Extracts a list of games and creates a pandas dataframe
Saves it into a csv for further processing. Single-threaded implementation.

Example: single_thread_preprocess_games('data/raw/sample.pgn', 'data/raw/sample.csv')
"""
def single_thread_preprocess_games(file_path, save_path, num_games=float('inf')):
    logging.info(f"Prepreoccesing PGN at {file_path} to save a csv at {save_path}")

    games = process_pgn(file_path, num_games)
    data = []

    for game in tqdm(games, desc="Processing games"):
        board = game.board()

        for move, node in zip(game.mainline_moves(), game.mainline()):
            board.push(move)
            encoded_board = encode_board(board)
            eval = extract_eval(board, node.comment)
            data.append((encoded_board, eval))

    df = pd.DataFrame(data, columns=['Board', 'Evaluation'])
    df.to_csv(save_path, index=False)
    
    logging.info(f"Processed {len(data)} board states.")


"""
Extracts a list of games and creates a pandas dataframe
Saves it into a csv for further processing. Multi-threaded implementation.

Example: multi_thread_preprocess_games('data/raw/sample.pgn', 'data/raw/sample.csv')
"""
def multi_thread_preprocess_games(file_path, save_boards_path, save_evals_path, num_games=float('inf')):
    logging.info(f"Prepreoccesing PGN at {file_path} to save a npy file at {save_boards_path} and {save_evals_path}")

    games = process_pgn(file_path, num_games)
    encoded_boards = []
    evaluations = []

    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_game, enumerate(games)), total=len(games), desc="Processing games"))

        for boards, evals in results:
            encoded_boards.extend(boards)
            evaluations.extend(evals)

    
    logging.info(f"Processed {len(encoded_boards)} board states.")

    encoded_boards = np.array(encoded_boards)
    evaluations = np.array(evaluations)

    np.save(save_boards_path, encoded_boards)
    np.save(save_evals_path, evaluations)

    # df = pd.DataFrame(data, columns=['Board', 'Evaluation'])
    # df.to_csv(save_path, index=False)


"""
This function processes only a single game which allows for multi-threaded execution without race conditions. 
Allow the multithreaded ThreadPoolExecutor to concurrently process multiple games.
"""
def process_game(game_tup):
    game_num, game = game_tup
    board: chess.Board = game.board()
    # print(game)
    # print(game.board())

    thread_id = threading.get_ident()
    logging.info(f"Process {thread_id} is processing game {game_num}")

    encoded_boards = []
    boards_to_analyze = []

    for move in game.mainline_moves():
        board.push(move)
        encoded_board = encode_board(board.copy())
        encoded_boards.append(encoded_board)
        boards_to_analyze.append(board.copy())

    stockfish_evaluations = get_stockfish_eval(boards_to_analyze)

    return encoded_boards, stockfish_evaluations

"""
Made obsolete, decided to just re-evaluate everything regardless of if it already has evals.
"""
# def extract_eval(label, stockfish_eval=None):
#     if "#" in label:
#         if label[1] == "-": # Checkmate for black
#             return -1000
#         else:
#             return 1000

#     eval = re.search(r'\[%eval (.*?)\]', label)

#     if eval != None:
#         return float(eval.group(1))
#     else:
#         return stockfish_eval





if __name__ == "__main__":
    file_path = "data/raw/my_games.pgn"
    # file_path = "data/raw/may_2023_database.pgn"
    # save_path_boards = "data/processed/10000_games_boards.npy"
    # save_path_evals = "data/processed/10000_games_evals.npy"
    save_path_boards = "data/processed/my_games_boards.npy"
    save_path_evals = "data/processed/my_games_evals.npy"

    # Uncomment to run the function
    # multi_thread_preprocess_games(file_path, save_path_boards, save_path_evals, num_games=1000)
    multi_thread_preprocess_games(file_path, save_path_boards, save_path_evals, num_games=100000)
