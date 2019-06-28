# general imports
import os
import copy
import logging
from categorical.helper.profiling import timing
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

model_name = os.path.split(os.path.dirname(__file__))[1]
dir_name = model_name

# helper imports

# Network
import categorical.model.MLP_LSTM as MLP_LSTM
# Data
from categorical.Motionsense.motionsense import Loader

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])

##############################################################################################
# Parameters
##############################################################################################



class HyperParameters(MLP_LSTM.NNParameters):
    """ HyperParamters for the neural network

    """

    def __init__(self, model_name, loglevel, parent_name=None):
        # Process parameter
        super().__init__(model_name, dir_name, loglevel, parent_name)
        # Loader
        loader = Loader()
        # # Data
        self.colum_names = copy.deepcopy(loader.input_signal_types)
        self.labels = copy.deepcopy(loader.labels)
        self.categorizations = copy.deepcopy(loader.classes)

        # Hyperparameter
        self.label = "subject"  # chose gender,subject or actvity
        if self.label == "gender":
            self.n_timesteps = 300
        elif self.label == "subject":
            self.n_timesteps = 600
        else:
            self.label = "activity"
            self.n_timesteps = 300
        self.label_categories = self.categorizations[self.label]
        # # Training (Hyperparameters)
        self.batch_size = 100
        self.epochs = 50
        # END Hyperparameter

        del loader


##############################################################################################
# Neural Network
# API
##############################################################################################


class DeepLearning(MLP_LSTM.NNDefinition):
    def __init__(self, hyperparameter):
        super().__init__(hyperparameter)
        self.parameter: HyperParameters = hyperparameter

    @timing
    def load_data(
        self
    ) -> Tuple[
        Tuple[Union[DataFrame, Series, ndarray,csr_matrix], ...],
        Tuple[Union[DataFrame, Series, ndarray,csr_matrix], ...],
        Tuple[Union[DataFrame, Series, ndarray,csr_matrix], ...],
    ]:
        # Import data
        loader = Loader()
        x_train, y_train, x_valid, y_valid, df_list, label_df = loader.motion_data(
            self.parameter.label, self.parameter.n_timesteps
        )

        train_data = x_train, y_train
        valid_data = x_valid, y_valid
        test_data = pd.DataFrame(), pd.DataFrame()
        self.train_data = train_data
        self.test_data = test_data
        self.validation_data = valid_data
        return train_data,  valid_data , test_data

    def setup_and_train_network(self):
        # Data
        # # Loading
        train_data, valid_data, test_data = self.load_data()

        # Model
        # # Definition
        model = self.define_model(train_data[0].shape, train_data[1].shape)

        # # Training
        model, final_metrics, label_vectors, training_history = self.train_network(epochs=self.parameter.epochs)

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
    # Train model
    neural_network.setup_and_train_network()
    # Do more stuff
    #neural_network.train_network(100)
    logger.debug("Finished")
