import chess
from data_preparation import squareNotation
from search_algorithm import simple_move
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/move", methods=["POST"])
def make_move():
    board_fen = request.json['board']
    model_name = request.json['model']

    model_path = "data/models/" + model_name + ".h5"
    model = load_model(model_path)
    board = chess.Board(fen=board_fen)

    move = simple_move(board, model, board.turn)

    return jsonify(success=True, uci=move.uci(), game_over=board.is_game_over())

# if __name__ == "__main__":
#     app.run()