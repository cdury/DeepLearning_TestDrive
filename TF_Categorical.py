# API - a) can run stand alone          (once per run)
#       b) gets called from HyperOpt.py (several times per run)
#
#     - Needs to import NeuralNetwork py-file with:
#         Functions:
#           x_train, y_train, x_test, y_test = data(hyperparameter)
#           model, placeholder = define_graph(*,hyperparameter)
#           train_losses, train_accuracies, test_losses, test_accuracies = train_graph_weights(*,hyperparameter)
#         NeuralNetwork Output:
#           Categorical
#         NeuralNetwork Framework:
#           Tensorflow 1.13
__version__="1.0"
from networks_tf import LSTM_TF_1Layer_HAR as NeuralNetwork
from helper_nn import helper_encoding as encoding

import os
import gc
import numpy as np
from typing import Any, Dict, Tuple, Union
from tensorflow import Tensor, Operation

import tensorflow as tf


# Hyperopt API
class HyperParameters:
    """ HyperParamters for the neural network

    """

    def __init__(self):
        # Init
        self.tid = 0  # Trial ID when using hyperopt
        # Build Modell
        self.n_hidden = 4  # Num of features in the first hidden layer
        self.n_classes = 6  # Number of classification classes
        # Optimizer
        self.learning_rate = 0.025
        self.lambda_loss_amount = 0.000191
        # Data
        self.colums_to_use = [0, 1]  # List of data columns to be used
        # Learn
        self.batch_size = 3000
        self.epochs = 100
        # Storage
        self.parent_name = ""  # Name of the optimization run when using hyperopt
        self.model_name = "LSTM_1_Layer"
        self.tb_suffix = os.path.splitext(os.path.basename(__file__))[0] +"_"+ self.model_name
        # Display
        self.display_iter = 30000

    def run_and_get_error(self) -> (Tuple[float, float, float, float], Dict[str, str]):
        """This function should call your tensorflow etc
        code which does the whole training procedure and returns
        the losses.  Lower is better!

        :return:
        """
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        gc.collect()
        loss_acc, additional_dict = neural_net(self)
        gc.collect()

        return loss_acc, additional_dict


hyperparameters = HyperParameters()


##############################################################################################
# Free to code
##############################################################################################
def data(hyperparameter: HyperParameters) -> (Any, Any):
    """ Load data into a train set and a test set

    :param hyperparameter:
    :return: train set and test set
    """

    x_train, y_train, x_test, y_test = NeuralNetwork.data(hyperparameter.colums_to_use)

    # We extract the mean and the standard defiation as features for our classificator.
    feature_0 = np.mean(x_train, axis=1)
    feature_1 = np.std(x_train, axis=1)
    f_c_train = np.concatenate((feature_0, feature_1), axis=1)

    feature_0_t = np.mean(x_test, axis=1)
    feature_1_t = np.std(x_test, axis=1)
    f_c_test = np.concatenate((feature_0_t, feature_1_t), axis=1)

    train_data = x_train, y_train, f_c_train
    test_data = x_test, y_test, f_c_test
    return train_data, test_data


def create_model(
    train_dimensions, hyperparameter: HyperParameters
) -> (Tuple[Union[Tensor, Operation]], Tuple[Tensor]):
    """ Define Graph with optimizer

    :param train_dimensions: Shape of the (training)data
    :param hyperparameter:
    :return: tensorflow model parts and a set of placeholder
    """
    # # Parameter
    n_input = train_dimensions[2]  # up to 9 input parameters per timestep
    n_steps = train_dimensions[1]  # 128 timesteps per series
    lambda_loss_amount = hyperparameter.lambda_loss_amount
    model, placeholder = NeuralNetwork.define_graph(
        n_input,
        hyperparameter.n_hidden,
        hyperparameter.n_classes,
        n_steps,
        hyperparameter.learning_rate,
        lambda_loss_amount,
    )
    return model, placeholder


def train_model(
    sess, model, placeholder, train_data, test_data, hyperparameter: HyperParameters
) -> (Any, Any):
    """

    :param sess: TF session
    :param model: TF model parts
    :param placeholder: Placeholder
    :param train_data: Training data set
    :param test_data:  Test data set
    :param hyperparameter:
    :return: losses and accuracy for train and test
    """
    # parameter entpacken
    x_train, y_train, f_c = train_data

    # # Training
    # Up to now we only defined the graph of our network, now we start a tensorflow session to acutally perform
    # the optimization.
    training_data_count = x_train.shape[0]  # number of measurement series

    # Launch Training
    train_losses, train_accuracies, test_losses, test_accuracies = NeuralNetwork.train_graph_weights(
        model,
        placeholder,
        sess,
        train_data,
        test_data,
        hyperparameter.batch_size,
        training_data_count * hyperparameter.epochs,
        hyperparameter.display_iter,
        hyperparameter.n_classes,
        hyperparameter.tb_suffix,
    )

    return train_losses, train_accuracies, test_losses, test_accuracies


def trained_model_stats(
    sess, model, placeholder, test_data, hyperparameter: HyperParameters
):
    """ Evaluate trained modell with test data

    :param sess: TF session
    :param model: TF model parts
    :param placeholder: Placeholder
    :param test_data:  Test data set
    :param hyperparameter:
    :return:
    """
    # parameter entpacken
    summ, pred, optimizer, cost, accuracy = model
    x, y, aux_obs = placeholder
    x_test, y_test, f_c_t = test_data

    # Accuracy for test data
    one_hot_predictions, accuracy, final_loss = sess.run(
        [pred, accuracy, cost],
        feed_dict={
            x: x_test,
            y: encoding.one_hot(y_test, hyperparameter.n_classes),
            aux_obs: f_c_t,
        },
    )
    print(
        "FINAL RESULT: "
        + "Batch Loss = {}".format(final_loss)
        + ", Accuracy = {}".format(accuracy)
    )

    return final_loss, accuracy


def neural_net(
    hyperparameter: HyperParameters
) -> (Tuple[float, float, float, float], Dict[str, str]):
    """ Run Neural Net

    :param hyperparameter:
    :return:
    """
    # Init
    additional_dict = {}
    # START
    # Load & Preprocess data
    train_data, test_data = data(hyperparameter)
    x_train, y_train, f_c = train_data
    x_test, y_test, f_c_t = test_data

    with tf.Graph().as_default():
        # Setup model
        model, placeholder = create_model(x_train.shape, hyperparameter)

        # Launch the graph & Training
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            # Train model
            train_losses, train_accuracies, test_losses, test_accuracies = train_model(
                sess, model, placeholder, train_data, test_data, hyperparameter
            )

            # Characteristics of trained model
            final_loss, accuracy = trained_model_stats(
                sess, model, placeholder, test_data, hyperparameter
            )
            test_losses.append(final_loss)
            test_accuracies.append(accuracy)
            loss_acc = (train_losses[-1], train_accuracies[-1], final_loss, accuracy)
    tf.reset_default_graph()

    # Save trained model
    additional_dict["stored_model"] = hyperparameter.model_name
    # DONE

    return loss_acc, additional_dict


def main() -> None:
    """ Calls neural_net() if started in terminal

    :return:
    """
    # This is where you do all the normal training code etc
    # Some initial set up if necessary
    # Do the actual work
    loss_acc, additional_dict = neural_net(hyperparameters)
    train_loss, train_accuracy, test_loss, test_accuracy = loss_acc
    # Evaluate trained model
    # # Further advanced evaluation
    # # Plots

    return None


if __name__ == "__main__":
    main()
