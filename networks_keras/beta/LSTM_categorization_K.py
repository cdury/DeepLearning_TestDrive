# Import whole packages
import os

import numpy as np
import pandas as pd

# Import Classes and Functions
from random import shuffle

# from tensorflow import Tensor, Operation
from tensorflow.python.keras.layers import Input, Dense, LSTM
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.callbacks import EarlyStopping

# Helper functions

# pathes
HOME_PATH = os.getcwd()
TB_PATH = os.path.join(HOME_PATH, "tensorboard")
DATA_PATH = os.path.join(HOME_PATH, "data")
MODEL_PATH = os.path.join(HOME_PATH, "models")
DATASET_PATH = os.path.join(DATA_PATH, "UCI_HAR_Dataset")

# tensorboard log
TRAIN = "train/"
TEST = "test/"
TB_NAME = "Verlauf_direkt"

# Logging
os.environ[
    "TF_CPP_MIN_LOG_LEVEL"
] = "0"  # 0: All Msg 1: No INFO 2: No INFO & WARNING 3: No INFO, WARNING & ERROR

# Printing
# If you would like to turn of scientific notation, the following line can be used:
np.set_printoptions(suppress=True)

# Check Versions
with open("print_imported_versions.py") as f:
    code = compile(f.read(), "print_imported_versions.py", "exec")
    exec(code)

# Labels of input collumns
INPUT_SIGNAL_TYPES = [
    "body_acc_x_",
    "body_acc_y_",
    "body_acc_z_",
    "body_gyro_x_",
    "body_gyro_y_",
    "body_gyro_z_",
    "total_acc_x_",
    "total_acc_y_",
    "total_acc_z_",
]

# Output classes to learn how to classify
LABELS = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]


def data(colums_to_use):
    # Load "X" (the neural network's training and testing inputs)
    def load_X(X_signals_paths):
        X_signals = []

        for signal_type_path in X_signals_paths:
            file = open(signal_type_path, "r")
            # Read dataset from disk, dealing with text files' syntax
            X_signals.append(
                [
                    np.array(serie, dtype=np.float32)
                    for serie in [
                        row.replace("  ", " ").strip().split(" ") for row in file
                    ]
                ]
            )
            file.close()

        return np.transpose(np.array(X_signals), (1, 2, 0))

    # Load "y" (the neural network's training and testing outputs)
    def load_y(y_path):
        file = open(y_path, "r")
        # Read dataset from disk, dealing with text file's syntax
        y_ = np.array(
            [
                elem
                for elem in [row.replace("  ", " ").strip().split(" ") for row in file]
            ],
            dtype=np.int32,
        )
        file.close()

        # Substract 1 to each output class for friendly 0-based indexing
        return y_ - 1

    # print("We use the features", [INPUT_SIGNAL_TYPES[i] for i in colums_to_use])

    X_train_signals_paths = [
        os.path.join(DATASET_PATH, TRAIN, "Inertial Signals", signal + "train.txt")
        for signal in INPUT_SIGNAL_TYPES
    ]
    X_test_signals_paths = [
        os.path.join(DATASET_PATH, TEST, "Inertial Signals", signal + "test.txt")
        for signal in INPUT_SIGNAL_TYPES
    ]
    X_train = load_X(X_train_signals_paths)[:, :, colums_to_use]
    X_test = load_X(X_test_signals_paths)[:, :, colums_to_use]

    y_train_path = os.path.join(DATASET_PATH, TRAIN, "y_train.txt")
    y_test_path = os.path.join(DATASET_PATH, TEST, "y_test.txt")
    y_train = load_y(y_train_path)
    y_test = load_y(y_test_path)

    idx_shuffel = np.arange(X_train.shape[0])
    shuffle(idx_shuffel)

    X_train = X_train[idx_shuffel, :, :]
    y_train = y_train[idx_shuffel]

    return X_train, y_train, X_test, y_test


def define_model(
    n_input,
    n_hidden,
    n_classes,
    n_steps,
    learning_rate,
    lambda_loss_amount,
    show_summary,
) -> Model:
    print("Build model...")

    # try using different optimizers and different optimizer configs

    # ToDo: Layer
    inputs = Input(shape=(n_steps, n_input))
    reshape_layer = Dense(n_hidden, activation="relu")(inputs)  # Reshape  to LSTM Input
    lstm_layer = LSTM(n_hidden)(reshape_layer)
    output = Dense(n_classes, activation="softmax")(lstm_layer)

    # ToDo: Tensor flow (Input, Output)
    # Graph input/output

    # This returns a tensor
    model = Model(inputs=inputs, outputs=output)

    # ToDo: Loss function part cross entropy

    # ToDo: Loss function part L2 regularization

    # ToDo: Loss function part auxiliary output

    # ToDo: Give summary
    if show_summary:
        model.summary()

    # ToDo: optimizer
    # As optimizer we use the Adam Optimizer
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(lr=learning_rate),
        metrics=["accuracy"],
    )

    return model


def train_graph_weights(
    model,
    callbacks,
    train_data,
    test_data,
    batch_size,
    epochs,
    display_iter,
    n_classes,
    TB_NAME,
    verbosity,
):
    X_train, y_train, f_c = train_data
    X_test, y_test, f_c_t = test_data
    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        verbose=verbosity,
        validation_data=(X_test, y_test),
    )
    print("fitted ...")

def main():
    # Startup Parameters
    colums_to_use = [0, 1]  # colums_to_use=[:]

    # Data
    # # ToDo: Loading and Group into train- & test-data
    X_train, y_train, X_test, y_test = data(colums_to_use)

    print(type(y_train))
    y_train = pd.get_dummies(y_train.flatten()).values.astype(np.float32)
    y_test = pd.get_dummies(y_test.reshape(len(y_test))).values.astype(np.float32)
    # # ToDo: Feature engineering
    # We extract the mean and the standard deviation as features for our classificator.
    feature_0 = np.mean(X_train, axis=1)
    feature_1 = np.std(X_train, axis=1)
    f_c = np.concatenate((feature_0, feature_1), axis=1)
    feature_0_t = np.mean(X_test, axis=1)
    feature_1_t = np.std(X_test, axis=1)
    f_c_t = np.concatenate((feature_0_t, feature_1_t), axis=1)

    # # ToDo: Group into train- , test- & validation-data
    # from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(
    #    X, y, test_size=0.3, random_state=42
    # )
    train_data = X_train, y_train, f_c
    test_data = X_test, y_test, f_c_t

    # Model
    # # ToDo: Parameter
    # # # Dimensions
    training_data_count = X_train.shape[0]
    n_steps = X_train.shape[1]  # 128 timesteps per series
    n_input = X_train.shape[2]  # 2 input parameters
    # # # Model
    n_hidden = 4  # Num of features in the first hidden layer
    n_classes = y_train.shape[1]  # Total classes
    # # # Optimizer
    learning_rate = 0.025
    # # # Loss function
    lambda_loss_amount = 0.000191
    # # # Training amount and scope
    epochs = 1 # 100
    training_iters = training_data_count * epochs
    batch_size = 3000
    # # # Informational
    display_iter = 30000  # To show test set accuracy during training
    # verbose=0 - No progress output (use with Juputer if you do not want output)
    # verbose=1 - Display progress bar, does not work well with Jupyter
    # verbose=2 - Summary progress output (use with Jupyter if you want to know the loss at each epoch)
    verbosity = 2
    show_summary = True
    show_graphs = True
    model_name = "UHR_Keras"
    nn_structure_filename = os.path.join(MODEL_PATH, model_name + ".png")

    # # ToDo: Defining
    # # # Structure
    # define_graph(n_input, n_hidden, n_classes, n_steps, learning_rate, lambda_loss_amount)
    model = define_model(
        n_input,
        n_hidden,
        n_classes,
        n_steps,
        learning_rate,
        lambda_loss_amount,
        show_summary,
    )

    if show_graphs:
        plot_model(
            model,
            to_file=nn_structure_filename,
            show_shapes=True,
            show_layer_names=True,
        )

    # summ, pred, optimizer, cost, accuracy = model
    # x, y, aux_obs = placeholder

    # # ToDo: Training
    # Up to now we only defined the graph of our network, now we start a tensorflow session to acutally perform
    # the optimization.
    # # # Configuration
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # # # Callbacks
    early_stopping = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-3,
        patience=5,
        verbose=1,
        mode="auto",
        restore_best_weights=True,
    )
    # # # Launch the graph & Training
    train_losses, train_accuracies, test_losses, test_accuracies = 0,0,0,0
    train_graph_weights(
        model,
        [],
        train_data,
        test_data,
        batch_size,
        epochs,
        display_iter,
        n_classes,
        TB_NAME,
        verbosity,
    )

    # ToDo: Final Accuracy
    # # Evaluation
    # Accuracy for test data

    evaluations = []
    model_evaluation = model.evaluate(X_test, y_test)
    model_evaluation = (
        model_evaluation if isinstance(model_evaluation, list) else [model_evaluation]
    )
    for name, value in zip(model.metrics_names, model_evaluation):
        evaluations.append(f"{name} = {value}")
    print("FINAL RESULT: " + ", ".join(evaluations))

    # # Prediciton
    # Accuracy for test data


if __name__ == "__main__":
    main()
