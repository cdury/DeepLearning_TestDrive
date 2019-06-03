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


#### Batch extractor
# We perform batchwise training, therefore we need a function that supplies the batches to the training algorithem:
def extract_batch_size(_train, step, batch_size):
    # Function to fetch a "batch_size" amount of data from "(X|y)_train" data.

    shape = list(_train.shape)
    shape[0] = batch_size
    batch_s = np.empty(shape)

    for i in range(batch_size):
        # Loop index
        index = ((step - 1) * batch_size + i) % len(_train)
        batch_s[i] = _train[index]

    return batch_s


#### One hot encoding
# To train with the classification we represent the labe as on hot encoded vector:
def one_hot(y_, n_classes):
    # Function to encode neural one-hot output labels from number indexes
    # e.g.:
    # one_hot(y_=[[5], [0], [3]], n_classes=6):
    #     return [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]

    y_ = y_.reshape(len(y_))
    return np.eye(n_classes)[np.array(y_, dtype=np.int32)]  # Returns FLOATS


# ToDo: Reshaping if necessary
def Reshaping(_X, _weights, _biases, n_input, n_steps):
    # Regardless of the number of input time series the input to the LSTM has to have  n_hidden dimensions
    # (To achive this we pass the input throug an neuron with RELU- Activation and weights
    #  _weights['hidden'], _biases['hidden'] ).
    """ Reshapes and scales _X, for scaling it uses _weights['hidden'] and _biases['hidden']"""
    # c#_X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    # c#_X = tf.reshape(_X, [-1, n_input])
    # new shape: (n_steps*batch_size, n_input)

    # ReLU activation, thanks to Yu Zhao for adding this improvement here:
    # c#_X = tf.nn.relu(tf.matmul(_X, _weights["hidden"]) + _biases["hidden"])
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    # c#_X = tf.split(_X, n_steps, 0)
    # new shape: n_steps * (batch_size, n_hidden)

    return _X


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
    # Classification loss function
    # soft_max_cost = (
    #    0
    # )  # c# tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pred))

    # We add upto two more terms to the loss function:
    # - l2 loss over the trainable weights
    # - (if needed) loss for the error in predicting the auxillary output

    # ToDo: Loss function part L2 regularization
    # Loss, optimizer and evaluation
    # l2 = (
    #    lambda_loss_amount * 1
    # )  # c# sum(        tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()    )
    # L2 loss prevents this overkill neural network to overfit the data
    # global_step = tf.Variable(0, trainable=False)
    # starter_learning_rate = learning_rate

    # ToDo: Loss function part auxiliary output
    # c# aux_cost = tf.nn.l2_loss(aux_out - aux_obs)
    cost = 0  # c# soft_max_cost + (1 / 300) * aux_cost + l2

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


def train_graph_weights(
    model,
    placeholder,
    sess,
    train_data,
    test_data,
    batch_size,
    training_iters,
    display_iter,
    n_classes,
    tb_suffix,
):
    summ, pred, optimizer, cost, accuracy = model
    x, y, aux_obs = placeholder
    X_train, y_train, f_c = train_data
    X_test, y_test, f_c_t = test_data
    # Parameter
    tb_path = os.path.join(TB_PATH, "train_" + tb_suffix)  # /tensorboard
    test_tb_path = os.path.join(TB_PATH, "test_" + tb_suffix)  # /tensorboard
    # print(tb_path, os.path.isdir(tb_path))
    # print(test_tb_path, os.path.isdir(test_tb_path))
    # ToDo: Tensorboard logging
    # To keep track of training's performance
    test_losses = []
    test_accuracies = []
    train_losses = []
    train_accuracies = []
    # c# train_writer = tf.summary.FileWriter(tb_path, sess.graph)
    # c# test_writer = tf.summary.FileWriter(test_tb_path)

    # -# init = tf.global_variables_initializer()
    # x# sess.run(init)

    # Perform Training steps with "batch_size" amount of example data at each loop
    model.fit(X_train, y_train, epochs=50, batch_size=32)  # starts training
    step = 1
    while step * batch_size <= training_iters:
        batch_xs = extract_batch_size(X_train, step, batch_size)
        batch_ys = one_hot(extract_batch_size(y_train, step, batch_size), n_classes)
        batch_aux = extract_batch_size(f_c, step, batch_size)

        # Fit training using batch data
        _, loss, acc = 0, 0, 0  # x# sess.run(
        # c#     [optimizer, cost, accuracy],
        # c#     feed_dict={x: batch_xs, y: batch_ys, aux_obs: batch_aux},
        # c#  )
        train_losses.append(loss)
        train_accuracies.append(acc)

        # Evaluate network only at some steps for faster training:
        if (
            (step * batch_size % display_iter == 0)
            or (step == 1)
            or (step * batch_size > training_iters)
        ):
            # To not spam console, show training accuracy/loss in this "if"
            print(
                "Training iter #"
                + str(step * batch_size)
                + ":   Batch Loss = "
                + "{:.6f}".format(loss)
                + ", Accuracy = {}".format(acc)
            )

            # c# s = sess.run(summ, feed_dict={x: batch_xs, y: batch_ys, aux_obs: batch_aux})
            # c# train_writer.add_summary(s, step)
            # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
            # c# loss, acc, s = sess.run(
            # c#    [cost, accuracy, summ],
            # c#    feed_dict={x: X_test, y: one_hot(y_test, n_classes), aux_obs: f_c_t},
            # c#)
            # c#test_writer.add_summary(s, step)
            print(
                "PERFORMANCE ON TEST SET: "
                + "Batch Loss = {}".format(loss)
                + ", Accuracy = {}".format(acc)
            )
            test_losses.append(loss)
            test_accuracies.append(acc)

        step += 1

    print("Training Finished!")

    return train_losses, train_accuracies, test_losses, test_accuracies


def main():
    # Startup Parameters

    # Data
    # # ToDo: Loading
    X, y, species = data()

    X_train, y_train = X, y
    X_test, y_test = [], []
    X_valid, y_valid = [], []

    # # ToDo: Feature engineering
    feature_train = []
    feature_test = []
    feature_valid = []

    # # ToDo: Group into train- , test- & validation-data
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    train_data = X_train, y_train, feature_train
    test_data = X_test, y_test, feature_test
    valid_data = X_valid, y_valid, feature_valid

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
    MODEL_NAME = "One_Input"
    nn_structure_filename = os.path.join(MODEL_PATH, MODEL_NAME + ".png")

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
    train_losses, train_accuracies, test_losses, test_accuracies = 0, 0, 0, 0

    # train_losses, train_accuracies, test_losses, test_accuracies = train_graph_weights(
    #     model,
    #     placeholder,
    #     sess,
    #     train_data,
    #     test_data,
    #     batch_size,
    #     training_iters,
    #     display_iter,
    #     n_classes,
    #     TB_NAME,
    # )

    # ToDo: Final Accuracy
    # # Evaluation
    # Accuracy for test data
    one_hot_predictions, accuracy, final_loss = 0, 0, 0
    # sess.run(
    #     [pred, accuracy, cost],
    #     feed_dict={x: X_test, y: one_hot(y_test, n_classes), aux_obs: f_c_t},
    # )
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
    # Accuracy might be a more easily understood error metric.  It is essentially a test score.  For all of the iris predictions,
    # what percent were correct?  The downside is it does not consider how confident the neural network was in each prediction.
    correct = accuracy_score(expected_classes, predict_classes)
    print(f"Accuracy: {correct}")


if __name__ == "__main__":
    main()
