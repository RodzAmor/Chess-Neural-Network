import chess
from data_preparation import squareNotation
from search_algorithm import simple_move, find_move
from tensorflow.keras.models import load_model
from tensorflow.python.eager import profiler
from minimal_model import MinimalChessModel
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import gc
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
    gc.collect()
    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss / 1024 / 1024 # Memory is in megabytes

    # profiler.start()

    board_fen = request.json['board']
    model_name = request.json['model']
    depth = request.json['depth']

    # model_path = "data/models/" + model_name
    model_path = "data/models/online.h5"
    model = MinimalChessModel(model_path)
    board = chess.Board(fen=board_fen)
    
    start_time = time.time()

    centipawn, move = model.find_move(board, depth=depth)
    
    execution_time = (time.time() - start_time) * 1000
    end_memory = process.memory_info().rss / 1024 / 1024
    memory_used = end_memory - start_memory

    # profiler_result = profiler.stop()
    # profiler.save('./logs', profiler_result)

    print(f"Execution time: {execution_time} ms")
    print(f"Memory Usage: {memory_used} mb")

    return jsonify(success=True, uci=move.uci(), game_over=board.is_game_over())

if __name__ == "__main__":
    app.run()