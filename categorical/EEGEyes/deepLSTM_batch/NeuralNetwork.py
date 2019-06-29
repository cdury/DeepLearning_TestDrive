# general imports
import os
import copy
import logging
from categorical._helper.profiling import timing
import numpy as np
import pandas as pd
import sklearn.metrics as skm
import matplotlib.pyplot as plt

# typing imports
from typing import Tuple, Union
from numpy import ndarray
from pandas import DataFrame, Series
from scipy.sparse import csr_matrix

# keras imports
from tensorflow.python.keras.api._v2 import keras

# # Backend
keras_backend = keras.backend
# # Layers
Input = keras.layers.Input
Dense = keras.layers.Dense
LSTM = keras.layers.LSTM
GaussianNoise = keras.layers.GaussianNoise
# # Model
Model = keras.models.Model
Sequential = keras.models.Sequential
# # Regularizers
l2 = keras.regularizers.l2
# # Optimizer
Adam = keras.optimizers.Adam
Adadelta = keras.optimizers.Adadelta
# # Utils
plot_model = keras.utils.plot_model

dir_name = os.path.split(os.path.split(os.path.dirname(__file__))[0])[1]
sub_dir_name = os.path.split(os.path.dirname(__file__))[1]
model_name = dir_name + "_" + sub_dir_name
dir_path = os.path.join(dir_name, sub_dir_name)

# _helper imports

# Network
import categorical._model.deepLSTM_batch as deepLSTM

# Data
from categorical.EEGEyes.eeg_eyes import Loader

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])

##############################################################################################
# Parameters
##############################################################################################


class HyperParameters(deepLSTM.NNParameters):
    """ HyperParamters for the neural network

    """

    def __init__(self, model_name, loglevel, parent_name=None):
        # Process parameter
        super().__init__(model_name, dir_path, loglevel, parent_name)
        # Loader
        loader = Loader()
        # # Data
        self.colum_names = copy.deepcopy(loader.input_signal_types)
        self.labels = copy.deepcopy(loader.label)
        self.categorizations = copy.deepcopy(loader.classes)

        # Hyperparameter
        self.label = self.labels
        self.classes = self.categorizations  # [self.label]
        # # Training (Hyperparameters)
        self.batch_size = 4
        self.epochs = 50
        # END Hyperparameter

        del loader


##############################################################################################
# Neural Network
# API
##############################################################################################


class DeepLearning(deepLSTM.NNDefinition):
    def __init__(self, hyperparameter):
        super().__init__(hyperparameter)
        self.parameter: HyperParameters = hyperparameter

    @timing
    def load_data(
        self
    ) -> Tuple[
        Tuple[Union[DataFrame, Series, ndarray, csr_matrix], ...],
        Tuple[Union[DataFrame, Series, ndarray, csr_matrix], ...],
        Tuple[Union[DataFrame, Series, ndarray, csr_matrix], ...],
        Tuple[Union[DataFrame, Series, ndarray, csr_matrix], ...],
    ]:
        # Import data
        loader = Loader()
        x_train, y_train, x_train_pred, y_train_pred, x_val, y_val, x_test, y_test = loader.eeg_data(
            self.parameter.batch_size
        )

        train_data = x_train, y_train
        train_data_pred = x_train_pred, y_train_pred
        test_data = x_test, y_test
        valid_data = x_val, y_val
        self.train_data = train_data
        self.test_data = test_data
        self.validation_data = valid_data
        return train_data, train_data_pred, test_data, valid_data

    def calc_categorical_accuracy(self, model, non_train_data, add_data=None):
        train_data_pred = add_data
        final_metrics = {}
        trainX_forPred, trainY_forPred = train_data_pred
        y_train_pred = model.predict(
            np.repeat(
                trainX_forPred, self.parameter.batch_size, axis=0
            )  ## need same input shape as for training and thus repeating 4 times
        )
        ## score
        final_metrics["accuracy"] = skm.accuracy_score(
            trainY_forPred[0, :, 0], np.where(y_train_pred[0, :, 0] > 0.8, 1, 0)
        )

        ## evaluate first batch (other are copies)
        plt.plot(trainY_forPred[0, :, 0], label="true")
        plt.plot(y_train_pred[0, :, 0], label="pred0", alpha=0.5)
        plt.plot(y_train_pred[0, :, 1], label="pred1", alpha=0.5)
        plt.legend()
        plt.show()

        testX, testY = non_train_data
        # evaulate test ste
        y_pred = model.predict(np.repeat(testX, self.parameter.batch_size, axis=0))
        ## score
        final_metrics["val_accuracy"] = skm.accuracy_score(
            testY[0, :, 0], np.where(y_pred[0, :, 0] > 0.8, 1, 0)
        )
        plt.plot(testY[0, :, 0], label="true")
        plt.plot(y_pred[0, :, 0], label="pred0", alpha=0.5)
        plt.plot(y_pred[0, :, 1], label="pred1", alpha=0.5)
        plt.legend()
        plt.show()
        return final_metrics

    def setup_and_train_network(self):
        # Data
        # # Loading
        train_data, train_data_pred, test_data, valid_data = self.load_data()

        # Model
        # # Definition
        model = self.define_model(train_data[0].shape, train_data[1].shape)

        # # Training
        model, final_metrics, label_vectors, training_history = self.train_network(
            epochs=self.parameter.epochs, accuracy_data=train_data_pred
        )

        return model, final_metrics, label_vectors, training_history


if __name__ == "__main__":
    # Information printout
    from tensorflow.python.client import device_lib

    logger.debug("ENVIRONMENT")
    for device in device_lib.list_local_devices():
        logger.debug(
            f"{device.name}:{device.device_type} with memory {device.memory_limit}"
        )
        logger.debug(f" {device.physical_device_desc}")

    # Define Hyperparameters
    hyperparameters = HyperParameters(model_name=model_name, loglevel=logging.DEBUG)
    # Setup Model
    neural_network = DeepLearning(hyperparameters)
    # Train _model
    neural_network.setup_and_train_network()
    # Do more stuff
    # neural_network.train_network(100)
    logger.debug("Finished")
