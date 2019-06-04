# API - a) can run stand alone          (once per run)
#       b) gets called from HyperOpt.py (several times per run)
#
#     - Needs to import NeuralNetwork py-file with:
#         Functions:
#           x_train, y_train, x_test, y_test = data(hyperparameter)
#           model  = define_model(*,hyperparameter)
#           train_losses, train_accuracies, test_losses, test_accuracies = train_network(*,hyperparameter)
#         NeuralNetwork Output:
#           Categorical
#         NeuralNetwork Framework:
#           Keras 2.24
#           Tensorflow 1.13
__version__="0.0alpha"
import os
import gc
import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple, Union
from pandas import DataFrame

from tensorflow import Tensor, Operation
from tensorflow.contrib.keras.api.keras.models import Sequential
import tensorflow.contrib.keras.api.keras.backend as K


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
        self.tb_suffix = ""

        # Display
        self.display_iter = 30000

    def runAndGetError(self):
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
    train_data, test_data = None, None
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
    model = Sequential()
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    other = None
    return model, other


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
    result = "model.fit(x_train, y_train, batch_size=100, epochs=100)"
    train_losses, train_accuracies, test_losses, test_accuracies = result
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
    final_loss, accuracy = None, None
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

    # Setup model
    model, placeholder = create_model(x_train.shape, hyperparameter)

    # Launch the graph & Training
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with K.get_session(config=config) as sess:
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
    K.clear_session()
    del model

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
    ## Further advanced evaluation
    ## Plots

    return None


if __name__ == "__main__":
    main()
