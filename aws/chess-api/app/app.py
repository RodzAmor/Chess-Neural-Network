import json
import numpy as np
import tensorflow as tf
import chess
from model import ChessModel

# model_file = '/opt/ml/models/test_chess_model'
# model = tf.keras.models.load_model(model_file)

def lambda_handler(event, context):
    try:
        httpMethod = event.get('httpMethod')
        # print("b")
        # print(httpMethod)
        if httpMethod != 'POST':
            return {
                'statusCode': 200,
                'headers': {
                    'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': '*'
                },
                'body': json.dumps(
                    {
                        'message': "At the moment, only POST requests are accepted.",
                    }
                )
            }
    except KeyError:
        return {
                'statusCode': 400,
                'headers': {
                    'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': '*'
                },
                'body': json.dumps(
                    {
                        'error': "Key error.",
                        'event': "event"
                    }
                )
            }
        # print('KeyError')
        # print(event)

    try:
        payload = json.loads(event.get('body'))
        board_fen = payload['board']
        model_name = payload['model']
        depth = payload['depth']

        board = chess.Board(board_fen)

        model_path = "/opt/ml/models/" + model_name
        chess_model = ChessModel(model_path)

        centipawn, move = chess_model.find_move(board, depth=depth)
        
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': '*'
            },
            'body': json.dumps(
                {
                    'success': True,
                    'uci': move.uci(),
                    'game_over': board.is_game_over(),
                    'centipawn': str(centipawn)
                }
            )
        }

    except Exception as e:
        return {
            'statusCode': 400,
            'headers': {
                'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': '*'
            },
            'body': json.dumps(
                {
                    'error': str(e),
                    'event': str(event),
                }
            )
        }