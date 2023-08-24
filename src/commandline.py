import chess
import random
import tensorflow as tf
from tensorflow.keras.models import load_model
from search_algorithm import find_best_move
# from preprocessing import preprocess_games
# from model import build_model, train_model, load_model
# from minimax import minimax


def play_game(model, depth, bot_is_white):
    board = chess.Board()

    while board.is_game_over() == False:
        if board.turn == bot_is_white:
            move = find_best_move(board, model, depth)
            
            board.push(move)
            
            print("Bot is playing", move)
        else:
            move = input("Type move in algebraic chess notation: ")

            try:
                move = board.parse_san(move)
                if move in board.legal_moves:
                    board.push(move)
                else:
                    print("Illegal move")
                    continue
            except:
                print("Invalid move. Here are the valid moves")
                print(board.legal_moves)
                continue
            

def main():
    bot_is_white = random.choice([True, False])

    if bot_is_white:
        print("Bot is playing at white")
    else:
        print("Bot is playing at black")

    tf.get_logger().setLevel('ERROR')
    
    model_path = "data/models/test.h5"
    model = load_model(model_path)
    depth = 3

    play_game(model, depth, bot_is_white)

if __name__ == "__main__":
    main()