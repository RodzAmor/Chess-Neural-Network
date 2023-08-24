import tensorflow as tf
import numpy as np
import wandb
from wandb.keras import WandbCallback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from data_preparation import load_data, split_data
# from keras.optimizers import Adam
# from tf.keras.optimizers.legacy import Adam

"""
Build the neural network.

*** NOTE *** 
Still deciding on the architecture so I will save this later.
"""
def build_model():
    # nn_model = Sequential([
    #     Conv2D(32, (3, 3), activation="relu", input_shape=(8, 8, 13)),
    #     MaxPooling2D((2, 2)),
    #     Conv2D(32, (3, 3), activation="relu"),
    #     Flatten(),
    #     Dense(64, activation="relu"),
    #     Dense(1)
    # ])
    nn_model = Sequential([
        Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(8, 8, 13)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(1, activation='linear')
    ])

    print(nn_model.summary())
    
    nn_model.compile(optimizer='adam', loss='mse')
    # model.compile(optimizer='adam', loss='mean_squared_error', metrics=[])
    
    return nn_model


"""
Loads the data after the preprocessing step

Moved to data_preparation.py
"""
# def load_data(board_path, evals_path):
#     # Load the encoded chessboards and evaluations
#     boards = np.load(board_path, allow_pickle=True)
#     evals = np.load(evals_path, allow_pickle=True)

#     return boards, evals

"""
Train the model with the training data
"""
def train_model(model, X_train, y_train, X_val, y_val, epochs=100000, batch_size=1024):
    trained_model = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    
    return trained_model
    

"""
Train the model and save the history throughout the process
"""
def train_model_history(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=1024):
    callbacks = [
        EarlyStopping(min_delta=0.001, patience=10, restore_best_weights=True, verbose=1),
        ModelCheckpoint("model_checkpoint.Keras", save_best_only=True),
        WandbCallback()
    ]
    history_trained_model = model.fit(x=X_train, y=y_train, epochs=epochs, verbose=1, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=callbacks)

    return history_trained_model

"""
Evaluates the accuracy and loss of the model based on the evals from the test data.
"""
def evaluate_model(model, X_test, y_test):
    loss = model.evaluate(X_test, y_test)
    print("Loss:", loss)
    return loss

if __name__ == "__main__":
    run = wandb.init(project="chess-model", config = {
        "epochs": 100,
        "batch_size": 1024,
    })

    board_path = "data/processed/10000_games_boards.npy"
    evals_path = "data/processed/10000_games_evals.npy"
    boards, evals = load_data(board_path, evals_path)


    # Splitting data (you can use the previously defined split_data function)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(boards, evals)

    # Building, training, and evaluating the model
    model = build_model()
    history = train_model_history(model, X_train, y_train, X_val, y_val)
    # evaluate_model(model, X_test, y_test)

    model.save('10000_games_model2.Keras')

    run.finish()
