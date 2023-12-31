import numpy as np
import math 
from sklearn.model_selection import train_test_split

def load_data(board_path, evals_path):
    boards = np.load(board_path, allow_pickle=True)
    evals = np.load(evals_path, allow_pickle=True)

    return boards, evals


"""
Splits data based on the test size. By default the test size is 15% of the data.
"""
def split_data(boards, evals, test_size=0.2, val_size=0.15):
    # Training and temp sets

    X_train, X_test, y_train, y_test = train_test_split(boards, evals, test_size=test_size)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=(val_size / (1 - test_size)))

    return X_train, X_val, X_test, y_train, y_val, y_test


def squareNotation(square, is_white):
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

    if is_white:
        row = 8 - math.floor(square / 8)
        col = square % 8
        return letters[col] + str(row)
    else:
        row = 1 + math.floor(square / 8)
        col = 8 - (square % 8)
        return letters[col] + str(row)

def main():
    board_path = "data/processed/sample_boards.npy"
    evals_path = "data/processed/sample_evals.npy"

    boards, evals = load_data(board_path, evals_path)

    X_train, X_test, y_train, y_test = train_test_split(boards, evals, test_size=0.15)

    print("Training set shape:", X_train.shape, y_train.shape)
    print("Test set shape:", X_test.shape, y_test.shape)


if __name__ == '__main__':
    main()