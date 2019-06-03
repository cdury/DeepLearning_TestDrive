# Import whole packages
import sys
import os

import numpy as np
import pandas as pd
import sklearn
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python import keras

# Import Classes and Functions
from random import shuffle
from typing import Tuple, Union, Any, Dict, List
from sklearn import metrics

# from tensorflow import Tensor, Operation
from tensorflow.python.keras.layers import Input, Dense, Activation
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.callbacks import EarlyStopping

# Helper functions
from helper_graphical import plot_data, plot_decision_boundary

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
] = "2"  # 0: All Msg 1: No INFO 2: No INFO & WARNING 3: No INFO, WARNING & ERROR

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


def data():
    df = pd.read_csv(
        "https://data.heatonresearch.com/data/t81-558/iris.csv", na_values=["NA", "?"]
    )

    # Convert to numpy - Classification
    x = df[["sepal_l", "sepal_w", "petal_l", "petal_w"]].values
    dummies = pd.get_dummies(df["species"])  # Classification
    species = dummies.columns
    y = dummies.values

    return x, y, species


def define_model(n_input, n_hidden, n_classes, learning_rate, show_summary) -> Model:

    # ToDo: Layer

    hidden_layer1 = Dense(n_hidden, activation="relu")  # Hidden 1
    hidden_layer2 = Dense(n_hidden / 2, activation="relu")  # Hidden 2
    output_layer = Dense(n_classes, activation="softmax")  # Output

    # ToDo: Tensor flow (Input, Output)
    # Graph input/output
    inputs = Input(shape=(n_input,))
    hidden1 = hidden_layer1(inputs)
    hidden2 = hidden_layer2(hidden1)
    output = output_layer(hidden2)
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
        optimizer=Adam(lr=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():
    # Startup Parameters

    # Data
    # # ToDo: Loading
    X, y, species = data()

    # # ToDo: Feature engineering

    # # ToDo: Group into train- , test- & validation-data
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Model
    # # ToDo: Parameter
    # # # Dimensions
    training_data_count = X_train.shape[1]
    # n_steps = len(X_train[0])  # 128 timesteps per series
    n_input = X_train.shape[1]  # 2 input parameters
    # # # Model
    n_hidden = 50  # Num of features in the first hidden layer
    n_classes = y_train.shape[1]  # Total classes
    # # # Optimizer
    learning_rate = 0.05
    # # # Loss function
    lambda_loss_amount = 0.000191
    # # # Training amount and scope
    epochs = 100
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
    model_name = "One_Input"
    nn_structure_filename = os.path.join(MODEL_PATH, model_name + ".png")

    # # ToDo: Defining
    # # # Structure
    # define_graph(n_input, n_hidden, n_classes, n_steps, learning_rate, lambda_loss_amount)
    model = define_model(n_input, n_hidden, n_classes, learning_rate, show_summary)

    # model.compile(loss="categorical_crossentropy", optimizer="adam")

    # model.fit(X, y, verbose=2, epochs=100)

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
    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        verbose=verbosity,
        callbacks=[early_stopping],
        validation_data=(X_test, y_test),
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
    prediction_test = model.predict(X_test)
    print(f"Shape: {prediction_test.shape}")
    # test_losses.append(final_loss)
    # test_accuracies.append(accuracy)
    predict_classes = np.argmax(prediction_test, axis=1)
    expected_classes = np.argmax(y_test, axis=1)
    print(f"Predictions: {predict_classes}")
    print(f"Expected: {expected_classes}")

    from sklearn.metrics import accuracy_score

    # Accuracy might be a more easily understood error metric.  It is essentially a test score.
    # For all of the iris predictions, what percent were correct?
    # The downside is it does not consider how confident the neural network was in each prediction.
    correct = accuracy_score(expected_classes, predict_classes)
    print(f"Accuracy: {correct}")


if __name__ == "__main__":
    main()
