import chess.pgn


# File used to experiment with basic commands and learn the libraries used
with open('data/raw/sample.pgn') as pgn_file:
    first_game = chess.pgn.read_game(pgn_file)
    
    for move in first_game.mainline():
        # print(move.eval())
        # print(move)
        pass

    moves = list(first_game.mainline())
    print(moves)