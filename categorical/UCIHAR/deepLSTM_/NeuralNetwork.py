# general imports
import os
import copy
import logging
from categorical._helper.profiling import timing
import numpy as np
import pandas as pd

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

# _helper imports
from categorical._helper.encoding import one_hot

dir_name = os.path.split(os.path.split(os.path.dirname(__file__))[0])[1]
sub_dir_name = os.path.split(os.path.dirname(__file__))[1]
model_name = dir_name + "_" + sub_dir_name
dir_path = os.path.join(dir_name, sub_dir_name)

# _helper imports

# Network
import categorical._model.deepLSTM_ as deepLSTM

# Data
from categorical.UCIHAR.uci_har import Loader

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
        # # Data
        self.colums_to_use = [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
        ]  # List of data columns to be used
        # # Training (Hyperparameters)
        self.batch_size = 0
        self.epochs = 2

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
    ]:
        # Import UCIHAR data
        loader = Loader()
        x_train, y_train, x_test, y_test = loader.uci_har_dataset_data(
            self.parameter.colums_to_use
        )

        # # Features
        # if len(x_train.shape) > 2:
        #     # Flatten Input
        #     x_train = x_train.reshape(x_train.shape[0], -1)
        #     x_test = x_test.reshape(x_test.shape[0], -1)
        #     # (or) Feature Extracion Input
        #     # feature_0 = np.mean(x_train, axis=1)
        #     # feature_1 = np.std(x_train, axis=1)
        #     # x_train = np.concatenate((feature_0, feature_1), axis=1)
        #     # feature_0 = np.mean(x_test, axis=1)
        #     # feature_1 = np.std(x_test, axis=1)
        #     # x_test = np.concatenate((feature_0, feature_1), axis=1)
        y_train = one_hot(y_train)
        y_test = one_hot(y_test)
        train_data = x_train, y_train
        valid_data = x_test, y_test
        test_data = np.ndarray([]), np.ndarray([])
        self.train_data = train_data
        self.test_data = test_data
        self.validation_data = valid_data
        return train_data, valid_data, test_data

    def setup_and_train_network(self):
        # Data
        # # Loading
        train_data, valid_data, test_data = self.load_data()

        # Model
        # # Definition
        model = self.define_model(train_data[0].shape, train_data[1].shape)

        # # Training
        model, final_metrics, label_vectors, training_history = self.train_network(
            epochs=self.parameter.epochs
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
