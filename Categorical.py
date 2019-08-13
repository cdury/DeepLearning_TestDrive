# API - a) can run stand alone          (once per run)
#       b) gets called from HyperOpt.py (several times per run)
#
#     - Needs to import NeuralNetwork py-file with:
#         Functions:
#           x_train, y_train, x_test, y_test = data(hyperparameter)
#           _model  = define_model(*,hyperparameter)
#           train_losses, train_accuracies, test_losses, test_accuracies = train_network(*,hyperparameter)
#         NeuralNetwork Output:
#           Categorical
#         NeuralNetwork Framework:
#           Keras 2.24
#           Tensorflow 1.13
__version__ = "1.0"
# general imports
import gc
import logging

# typing imports
from typing import Any, Dict, Tuple, Optional

# _helper import
from categorical._helper.graphical import category_evaluation

# _model import
# ToDo: Import specific neural network
#import categorical.BostonHousing.MLP.NeuralNetwork as Network
#import categorical.EEGEyes.deepLSTM_batch.NeuralNetwork as Network
#import categorical.MNIST.MLP.NeuralNetwork as Network
#import categorical.Motionsense.CNN1d.NeuralNetwork as Network
#import categorical.Motionsense.CNN1d_LSTM.NeuralNetwork as Network
#import categorical.Motionsense.CNN2d.NeuralNetwork as Network
#import categorical.Motionsense.MLP_LSTM.NeuralNetwork as Network
#import categorical.UCIHAR.MLP.NeuralNetwork as Network
#import categorical.UCIHAR.deepLSTM_.NeuralNetwork as Network
#import categorical.UCIHAR.vanillaLSTM.NeuralNetwork as Network
import categorical.SimpleChart.MLP.NeuralNetwork as Network


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
    neural_network = Network.DeepLearning(hyperparameter)

    # START
    # # _model
    model, final_metrics, label_vectors, history = (
        neural_network.setup_and_train_network()
    )
    print("Final metrics", final_metrics)
    # # loss_acc
    train_losses = history.history["loss"] if "loss" in history.history else [0]
    train_accuracies = (
        history.history["accuracy"] if "accuracy" in history.history else [0]
    )
    test_losses = history.history["val_loss"] if "val_loss" in history.history else [0]
    test_accs = (
        history.history["val_accuracy"] if "val_accuracy" in history.history else [0]
    )
    final_loss = final_metrics["loss"]
    accuracy = final_metrics["accuracy"]
    loss_acc = (train_losses[-1], train_accuracies[-1], final_loss, accuracy)

    # # additional_dict
    additional_dict = {}
    additional_dict["stored_model"] = neural_network.parameter.model_name
    classes = neural_network.parameter.classes
    additional_dict["classes"] = classes
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
    classes = additional_dict["classes"]
    given = additional_dict["given"]
    predictions = additional_dict["predictions"]

    # ToDo: Visulaisation of the results
    # Show Results
    # # Plots
    category_evaluation(len(classes), classes, given, predictions)

    return None


if __name__ == "__main__":
    # Define Hyperparameters
    # ToDo: Check loglevel
    hyperparameters = Network.HyperParameters(
        model_name=Network.model_name, loglevel=logging.DEBUG
    )
    run_and_evaluate(hyperparameters)
