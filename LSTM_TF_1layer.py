import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import metrics
from random import shuffle
import os
from typing import Tuple, Union
from tensorflow import Tensor, Operation

# pathes
HOME_PATH = r"D:\DEV\lokal\fraunhofer\\"
TB_PATH = r"D:\DEV\lokal\fraunhofer\tensorboard\\"
DATA_PATH = r"D:\DEV\lokal\fraunhofer\data\\"  # win10
DATASET_PATH = DATA_PATH + "UCI_HAR_Dataset/"

# tensorboard log
TRAIN = "train/"
TEST = "test/"

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
        DATASET_PATH + TRAIN + "Inertial Signals/" + signal + "train.txt"
        for signal in INPUT_SIGNAL_TYPES
    ]
    X_test_signals_paths = [
        DATASET_PATH + TEST + "Inertial Signals/" + signal + "test.txt"
        for signal in INPUT_SIGNAL_TYPES
    ]
    X_train = load_X(X_train_signals_paths)[:, :, colums_to_use]
    X_test = load_X(X_test_signals_paths)[:, :, colums_to_use]

    y_train_path = DATASET_PATH + TRAIN + "y_train.txt"
    y_test_path = DATASET_PATH + TEST + "y_test.txt"
    y_train = load_y(y_train_path)
    y_test = load_y(y_test_path)

    idx_shuffel = np.arange(X_train.shape[0])
    shuffle(idx_shuffel)

    X_train = X_train[idx_shuffel, :, :]
    y_train = y_train[idx_shuffel]

    return X_train, y_train, X_test, y_test


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


def Reshaping(_X, _weights, _biases, n_input, n_steps):
    # Regardless of the number of input time series the input to the LSTM has to have  n_hidden dimensions
    # (To achive this we pass the input throug an neuron with RELU- Activation and weights
    #  _weights['hidden'], _biases['hidden'] ).
    """ Reshapes and scales _X, for scaling it uses _weights['hidden'] and _biases['hidden']"""
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, n_input])
    # new shape: (n_steps*batch_size, n_input)

    # ReLU activation, thanks to Yu Zhao for adding this improvement here:
    _X = tf.nn.relu(tf.matmul(_X, _weights["hidden"]) + _biases["hidden"])
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, n_steps, 0)
    # new shape: n_steps * (batch_size, n_hidden)

    return _X


def define_graph(
    n_input, n_hidden, n_classes, n_steps, learning_rate, lambda_loss_amount
) -> (Tuple[Union[Tensor, Operation]], Tuple[Tensor]):
    # # LSTM Neural Network's internal structure
    # # #To define the network we do not only need the weights (and biases) for the input neuron
    # (which is passed to the Reshaping function), but also the weights of the output neurons.
    # We regard this weights as learnable parameter, hence they are represented as tensorflow variables
    # (initialized with  normal deistirbuted random variables).

    def LSTM_RNN_one_Layer(_X, _weights, _biases, n_hidden, n_input, n_steps):
        _X_reshaped = Reshaping(_X, _weights, _biases, n_input, n_steps)

        lstm_cell = tf.contrib.rnn.LSTMBlockFusedCell(n_hidden, forget_bias=1.0)
        outputs, state = lstm_cell(tf.stack(_X_reshaped), dtype=tf.float32)
        return tf.matmul(tf.unstack(outputs)[-1], _weights["out"]) + _biases["out"]

    # Graph weights
    weights_one_Layer = {
        "hidden": tf.Variable(tf.random_normal([n_input, n_hidden])),
        "out": tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0)),
    }
    biases_one_Layer = {
        "hidden": tf.Variable(tf.random_normal([n_hidden])),
        "out": tf.Variable(tf.random_normal([n_classes])),
    }
    # # Training Setup
    # Before we start training we have to
    # - define the nodes that tensorflow should use to fill in the data set (the placeholders)
    # - conecte the input placholder with the defined networks
    # - define a loss function (where the placeholder for the input is filled in)

    # Graph input/output
    x = tf.placeholder(tf.float32, [None, n_steps, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])
    aux_obs = tf.placeholder(tf.float32, [None, 4])
    placeholder = (x, y, aux_obs)

    weights = weights_one_Layer
    biases = biases_one_Layer
    aux_out = 0
    pred = LSTM_RNN_one_Layer(x, weights, biases, n_hidden, n_input, n_steps)
    # Classification loss function
    soft_max_cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pred)
    )

    # We add upto two more terms to the loss function:
    # - l2 loss over the trainable weights
    # - (if needed) loss for the error in predicting the auxillary output

    # Loss, optimizer and evaluation
    l2 = lambda_loss_amount * sum(
        tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
    )  # L2 loss prevents this overkill neural network to overfit the data
    # global_step = tf.Variable(0, trainable=False)
    # starter_learning_rate = learning_rate

    aux_cost = tf.nn.l2_loss(aux_out - aux_obs)
    cost = soft_max_cost + (1 / 300) * aux_cost + l2

    # For observing the training progress in tensorflow we track the costs and some additional statistics:
    tf.summary.scalar("l2", l2)
    tf.summary.scalar("aux_cost", aux_cost)
    tf.summary.scalar("soft_max_cost", soft_max_cost)
    tf.summary.histogram("aux_cost_hist", aux_out - aux_obs)
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar("loss", cost)
    tf.summary.scalar("acc", accuracy)
    tf.summary.histogram("W_hidden", weights["hidden"])
    # tf.summary.histogram("W_aux_out", weights["aux_out"])
    tf.summary.histogram("W_out", weights["out"])
    summ = tf.summary.merge_all()

    # As optimizer we use the Adam Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(
        cost
    )  # Adam Optimizer

    model = summ, pred, optimizer, cost, accuracy
    return model, placeholder


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

    # To keep track of training's performance
    test_losses = []
    test_accuracies = []
    train_losses = []
    train_accuracies = []
    train_writer = tf.summary.FileWriter(tb_path, sess.graph)
    test_writer = tf.summary.FileWriter(test_tb_path)

    init = tf.global_variables_initializer()
    sess.run(init)

    # Perform Training steps with "batch_size" amount of example data at each loop
    step = 1
    while step * batch_size <= training_iters:
        batch_xs = extract_batch_size(X_train, step, batch_size)
        batch_ys = one_hot(extract_batch_size(y_train, step, batch_size), n_classes)
        batch_aux = extract_batch_size(f_c, step, batch_size)

        # Fit training using batch data
        _, loss, acc = sess.run(
            [optimizer, cost, accuracy],
            feed_dict={x: batch_xs, y: batch_ys, aux_obs: batch_aux},
        )
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

            s = sess.run(summ, feed_dict={x: batch_xs, y: batch_ys, aux_obs: batch_aux})
            train_writer.add_summary(s, step)
            # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
            loss, acc, s = sess.run(
                [cost, accuracy, summ],
                feed_dict={x: X_test, y: one_hot(y_test, n_classes), aux_obs: f_c_t},
            )
            test_writer.add_summary(s, step)
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
    colums_to_use = [0, 1]  # colums_to_use=[:]

    # Data
    # # Loading
    X_train, y_train, X_test, y_test = data(colums_to_use)
    # # Features
    # We extract the mean and the standard deviation as features for our classificator.
    feature_0 = np.mean(X_train, axis=1)
    feature_1 = np.std(X_train, axis=1)
    f_c = np.concatenate((feature_0, feature_1), axis=1)
    feature_0_t = np.mean(X_test, axis=1)
    feature_1_t = np.std(X_test, axis=1)
    f_c_t = np.concatenate((feature_0_t, feature_1_t), axis=1)
    train_data = X_train, y_train, f_c
    test_data = X_test, y_test, f_c_t
    # Graph
    # # Parameter
    training_data_count = len(X_train)
    # test_data_count = len(X_test)
    n_steps = len(X_train[0])  # 128 timesteps per series
    n_input = len(X_train[0][0])  # 9 input parameters per timestep

    learning_rate = 0.025
    lambda_loss_amount = 0.000191
    training_iters = training_data_count * 100  # Loop 300 times on the dataset
    batch_size = 3000
    display_iter = 30000  # To show test set accuracy during training
    n_hidden = 4  # Num of features in the first hidden layer
    n_classes = 6  # Total classes

    model, placeholder = define_graph(
        n_input, n_hidden, n_classes, n_steps, learning_rate, lambda_loss_amount
    )
    summ, pred, optimizer, cost, accuracy = model
    x, y, aux_obs = placeholder

    # # Training
    # Up to now we only defined the graph of our network, now we start a tensorflow session to acutally perform
    # the optimization.
    # Launch the graph & Training
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        train_losses, train_accuracies, test_losses, test_accuracies = train_graph_weights(
            model,
            placeholder,
            sess,
            train_data,
            test_data,
            batch_size,
            training_iters,
            display_iter,
            n_classes,
            "Verlauf_52",
        )

        # Accuracy for test data
        one_hot_predictions, accuracy, final_loss = sess.run(
            [pred, accuracy, cost],
            feed_dict={x: X_test, y: one_hot(y_test, n_classes), aux_obs: f_c_t},
        )
        print(
            "FINAL RESULT: "
            + "Batch Loss = {}".format(final_loss)
            + ", Accuracy = {}".format(accuracy)
        )
        test_losses.append(final_loss)
        test_accuracies.append(accuracy)


if __name__ == "__main__":
    main()
