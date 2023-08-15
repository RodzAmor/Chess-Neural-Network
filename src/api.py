import chess
from data_preparation import squareNotation
from search_algorithm import simple_move, find_move
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import psutil
import time

app = Flask(__name__)
CORS(app)

"""
API Endpoint that returns a json of the move for the website.

I added code to measure the memory and ram used which can then be used
later to estimate the price paid to AWS lambda.
"""
@app.route("/move", methods=["POST"])
def make_move():
    # Measure memory usage and time
    # start_time = time.time()
    # process = psutil.Process(os.getpid())
    # start_memory = process.memory_info().rss / 1024 / 1024 # This will be in megabytes

    board_fen = request.json['board']
    model_name = request.json['model']

    model_path = "data/models/" + model_name + ".h5"
    model = load_model(model_path)
    board = chess.Board(fen=board_fen)

    centipawn, move = find_move(board, model, depth=3)
    # print(centipawn)

    # execution_time = (time.time() - start_time) * 1000
    # end_memory = process.memory_info().rss / 1024 / 1024
    # memory_used = end_memory - start_memory

    # print(f"Execution time: {execution_time} ms")
    # print(f"Memory Usage: {memory_used} mb")


    return jsonify(success=True, uci=move.uci(), game_over=board.is_game_over())

if __name__ == "__main__":
    app.run()