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
__version__ = "0.0alpha"
import os
import gc
import numpy as np
import pandas as pd
import tensorflow as tf

from typing import Any, Dict, Tuple, Union, Optional
from pandas import DataFrame
from tensorflow import Tensor, Operation
from tensorflow.contrib.keras.api.keras.models import Sequential
import tensorflow.contrib.keras.api.keras.backend as K

from helper_nn.helper_graphical import category_evaluation

# from networks_keras.SNN_K_nLayer_MNIST import SNNLayerN as NeuralNetwork
# from networks_keras.SNN_K_nLayer_MNIST import HyperParameters
# from networks_keras.SNN_K_nLayer_MNIST import model_name

# from networks_keras.SNN_K_nLayer_BOSTON import SNNLayerN as NeuralNetwork
# from networks_keras.SNN_K_nLayer_BOSTON import HyperParameters
# from networks_keras.SNN_K_nLayer_BOSTON import model_name

from networks_keras.SNN_K_nLayer_HAR import SNNLayerN as NeuralNetwork
from networks_keras.SNN_K_nLayer_HAR import HyperParameters
from networks_keras.SNN_K_nLayer_HAR import model_name

##############################################################################################
# Free to code
##############################################################################################
def neural_net(
    hyperparameter
) -> (Tuple[float, float, float, float], Dict[str, str], Any):
    """ Run Neural Net

    :param hyperparameter:
    :return:
    """
    # Init
    neural_network = NeuralNetwork(hyperparameter)

    # START
    # # model
    model, final_metrics, label_vectors, history = (
        neural_network.setup_and_train_network()
    )
    # # loss_acc
    train_losses = history.history["loss"] if "loss" in history.history else [0]
    train_accuracies = history.history["acc"] if "acc" in history.history else [0]
    test_losses = history.history["val_loss"] if "val_loss" in history.history else [0]
    test_accs = history.history["val_acc"] if "val_acc" in history.history else [0]
    final_loss = final_metrics["loss"]
    accuracy = final_metrics["acc"]
    loss_acc = (train_losses[-1], train_accuracies[-1], final_loss, accuracy)

    # # additional_dict
    additional_dict = {}
    additional_dict["stored_model"] = neural_network.parameter.model_name
    labels = neural_network.parameter.labels
    additional_dict["labels"] = labels
    predictions, given = label_vectors
    additional_dict["given"] = given
    additional_dict["predictions"] = predictions

    # DONE
    del neural_network
    return loss_acc, additional_dict, model


def run_and_get_error(
    hyperparameter, return_model=True
) -> (Tuple[float, float, float, float], Dict[str, str], Optional[Any]):
    """This function should call your tensorflow etc
    code which does the whole training procedure and returns
    the losses.  Lower is better!

    :return:
    """
    gc.collect()
    loss_acc, additional_dict, model = neural_net(hyperparameter)

    if return_model:
        return loss_acc, additional_dict, model
    else:
        del model
        gc.collect()
        return loss_acc, additional_dict


def run_and_evaluate(hyperparameter) -> None:
    """ Calls neural_net() if started in terminal

    :return:
    """
    # This is where you do all the normal training code etc
    # Some initial set up if necessary
    # Do the actual work
    loss_acc, additional_dict, model = run_and_get_error(
        hyperparameter, return_model=True
    )
    # Unpack results
    train_loss, train_accuracy, test_loss, test_accuracy = loss_acc
    labels = additional_dict["labels"]
    given = additional_dict["given"]
    predictions = additional_dict["predictions"]
    # Show Results
    # # Plots
    category_evaluation(len(labels), labels, given, predictions)

    return None


if __name__ == "__main__":
    hyperparameters = HyperParameters(model_name=model_name, loglevel=3)
    run_and_evaluate(hyperparameters)
