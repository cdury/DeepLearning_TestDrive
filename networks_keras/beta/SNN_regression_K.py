# Import whole packages
import os

import numpy as np
import matplotlib.pyplot as plt

# Import Classes and Functions

# from tensorflow import Tensor, Operation
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras.callbacks import EarlyStopping

# Helper functions
from helper_keras.helper_graphical import plot_data, plot_decision_boundary

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


def define_model(n_input, n_hidden, learning_rate, show_summary) -> Model:
    # ToDo: Layer
    hidden_layer1 = Dense(n_hidden, activation="tanh", name="FirstLayer")
    hidden_layer2 = Dense(n_hidden, activation="tanh", name="SecondLayer")
    output_layer = Dense(1, activation="sigmoid", name="OutputLayer")

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
    model.compile(Adam(lr=learning_rate), "binary_crossentropy", metrics=["accuracy"])
    return model


def main():
    # Startup Parameters

    # Data
    # # ToDo: Loading
    from sklearn.datasets import make_circles

    X, y = make_circles(n_samples=1000, factor=0.6, noise=0.1, random_state=42)
    pl = plot_data(plt, X, y)
    pl.show()

    # # ToDo: Feature engineering

    # # ToDo: Group into train- , test- & validation-data
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Model
    # # ToDo: Parameter
    # # # Dimensions
    training_data_count = len(X_train)
    n_input = len(X_train[0])  # 2 input parameters
    # # # Model
    n_hidden = 4  # Num of features in the first hidden layer
    # # # Optimizer
    learning_rate = 0.05
    # # # Loss function
    # # # Training amount and scope
    epochs = 100
    # # # Informational
    verbosity = 1
    show_summary = True
    show_graphs = True
    model_name = "One_Input"
    nn_structure_filename = os.path.join(MODEL_PATH, model_name + ".png")

    # # ToDo: Defining
    # # # Structure
    model = define_model(n_input, n_hidden, learning_rate, show_summary)
    if show_graphs:
        plot_model(
            model,
            to_file=nn_structure_filename,
            show_shapes=True,
            show_layer_names=True,
        )

    # # ToDo: Training
    # Up to now we only defined the graph of our network, now we start a tensorflow session to acutally perform
    # the optimization.
    # # # Configuration
    # # # Callbacks
    early_stopping = EarlyStopping(monitor="val_accuracy", patience=5, mode="max")
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
    for name, value in zip(model.metrics_names, model.evaluate(X_test, y_test)):
        evaluations.append(f"{name} = {value}")
    print("FINAL RESULT: " + ", ".join(evaluations))

    # # Prediciton

    if show_graphs:
        plot_decision_boundary(model, X, y).show()


if __name__ == "__main__":
    # Check Versions
    with open("print_imported_versions.py") as f:
        code = compile(f.read(), "print_imported_versions.py", "exec")
        exec(code)
    # Run
    main()
