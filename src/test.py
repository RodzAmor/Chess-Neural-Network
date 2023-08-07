import chess.pgn

with open('sample.pgn') as pgn_file:
    first_game = chess.pgn.read_game(pgn_file)
    print(first_game.headers['Event'])

    moves = list(first_game.mainline_moves())
    print(moves)