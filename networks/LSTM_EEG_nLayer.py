# general imports
import os, sys
import time
import logging
import numpy as np
import pandas as pd
import sklearn.metrics as skm
import tensorflow as tf
import matplotlib.pyplot as plt

# typing imports
from typing import Tuple, List, Any, Union, Optional
from numpy import ndarray
from pandas import DataFrame, Series

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

model_name = os.path.splitext(os.path.basename(__file__))[0]

# helper imports
from helper.helper_encoding import one_hot

##############################################################################################
# Parameters
##############################################################################################
# Data
from helper.helper_load_data import eeg_data as data
from helper.helper_load_data import EEG_INPUT_SIGNAL_TYPES as INPUT_SIGNAL_TYPES
from helper.helper_load_data import EEG_LABELS as LABELS

# Network
from networks.Base_Supervised_Categorical import BaseParameters, BaseNN, timing

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])


class HyperParameters(BaseParameters):
    """ HyperParamters for the neural network

    """

    def __init__(self, model_name, loglevel, parent_name=None):
        # Process parameter
        super().__init__(model_name, loglevel, parent_name)
        # Filesystem & Paths
        # # data
        self.data_dir = self.data_path

        # Hyperparameter
        # # Data
        self.colum_names = INPUT_SIGNAL_TYPES
        self.labels = LABELS  # Labels of th categorizations
        # # Modell (Hyperparamters)
        self.n_hidden_1 = 100  # Num of hidden features in the first lstm layer
        self.hidden_1_noise = 1.5
        self.n_hidden_2 = 50
        self.hidden_2_noise = 0.5
        self.dropout = 0.2
        self.init_kernel = "random_normal"  # "he_normal", 'random_normal'
        self.activation_hidden = "relu"
        self.activation_output = "softmax"

        # # Optimizer (Hyperparamters)
        self.learning_rate = 1.0
        self.rho = 0.95
        self.decay = 0.0
        self.lambda_loss_amount = 0.005
        self.metric = "accuracy"

        # END Hyperparameter


##############################################################################################
# Neural Network
# API
##############################################################################################


class LSTMLayerN(BaseNN):
    def __init__(self, hyperparameter):
        super().__init__(hyperparameter)
        self.parameter: HyperParameters = hyperparameter

    @timing
    def load_data(
        self
    ) -> Tuple[
        Tuple[Union[DataFrame, Series, ndarray], ...],
        Tuple[Union[DataFrame, Series, ndarray], ...],
        Tuple[Union[DataFrame, Series, ndarray], ...],
        Tuple[Union[DataFrame, Series, ndarray], ...],
    ]:
        # Import Boston data
        x_train, y_train, x_train_pred, y_train_pred, x_val, y_val, x_test, y_test = data(
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

    @timing
    def define_model(self, input_shape, output_shape) -> Model:
        # Input (number of inputs)
        n_timesteps = input_shape[1]
        n_input = input_shape[2]
        self.parameter.n_input = (n_timesteps, n_input)
        # Output (number of classes)
        n_classes = output_shape[2]
        self.parameter.n_classes = n_classes

        # Start defining the input tensor:
        model = Sequential()
        model.add(
            LSTM(
                self.parameter.n_hidden_1
                # , input_shape=(None, X.shape[1])
                ,
                batch_input_shape=(self.parameter.batch_size, n_timesteps, n_input),
                stateful=True,
                return_sequences=True,
                recurrent_regularizer=l2(self.parameter.lambda_loss_amount)
                # , kernel_regularizer = regularizers.l1(0.01)
                # , activity_regularizer = regularizers.l2(0.01)
                ,
                dropout=self.parameter.dropout,  ## applied to input
            )
        )
        model.add(GaussianNoise(self.parameter.hidden_1_noise))
        model.add(
            LSTM(
                self.parameter.n_hidden_2,
                stateful=True,
                return_sequences=True,
                recurrent_regularizer=l2(self.parameter.lambda_loss_amount),
                dropout=self.parameter.dropout,  ## applied to input
            )
        )
        model.add(GaussianNoise(self.parameter.hidden_2_noise))
        model.add(
            Dense(self.parameter.n_classes, activation=self.parameter.activation_output)
        )

        # Define the loss function
        # loss_fn = lambda y_true, y_pred: tf.nn.softmax_cross_entropy_with_logits(
        #     logits=y_pred, labels=y_true
        # )
        loss_fn = "categorical_crossentropy"

        # Define the optimizer
        optimizer_fn = Adadelta(
            lr=self.parameter.learning_rate,
            rho=self.parameter.rho,
            epsilon=None,
            decay=self.parameter.decay,
        )
        # optimizer_fn = tf.train.AdamOptimizer(
        #     learning_rate=self.parameter.learning_rate
        # )

        # put all components together
        model.compile(
            loss=loss_fn, optimizer=optimizer_fn, metrics=[self.parameter.metric]
        )
        if self.parameter.loglevel <= logging.INFO:
            model.summary()
        if self.parameter.show_graphs:
            plot_model(
                model,
                to_file=os.path.join(
                    self.parameter.model_dir, self.parameter.model_name + ".png"
                ),
                show_shapes=True,
                show_layer_names=True,
            )
        self.model = model
        return model

    @timing
    def train_model(self, model, train_data, validation_data,initial_epoch=0):
        # Data
        x_train, y_train = train_data
        x_val, y_val = validation_data

        # Callbacks
        tensorboard, checkpoint, earlyterm = self.get_callbacks()
        used_callbacks = [tensorboard]

        # Training
        if self.parameter.batch_size > 0:
            history = model.fit(
                x_train,
                y_train,
                validation_data=(x_val, y_val),
                batch_size=self.parameter.batch_size,
                initial_epoch=initial_epoch,
                epochs=self.parameter.epochs,
                verbose=self.parameter.fitting_verbosity,
                callbacks=used_callbacks,
                shuffle=False,
            )
        else:
            history = model.fit(
                x_train,
                y_train,
                validation_data=(x_val, y_val),
                initial_epoch=initial_epoch,
                epochs=self.parameter.epochs,
                verbose=self.parameter.fitting_verbosity,
                callbacks=used_callbacks,
                shuffle=False,
            )
        self.parameter.trained_epochs = history.epoch[-1]+1
        model.reset_states()  # because of model->stateful=True
        logger.info("Training Finished!")
        return history

    def calc_categorical_accuracy(self, model, train_data_pred, test):
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

        testX, testY = test
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

    def train_network(self, epochs=0):
        initial_epoch = self.parameter.trained_epochs
        train_data = self.train_data
        valid_data = self.validation_data
        test_data = self.test_data
        model = self.model
        self.parameter.epochs = epochs
        # # Training
        training_history = self.train_model(model, train_data, valid_data, initial_epoch=initial_epoch)

        # # Calculate accuracy
        final_metrics = self.calc_categorical_accuracy(
            model, test_data, test_data
        )

        # # Calulate prediction
        predictions, given = self.is_vs_should_categorical(model, test_data)
        label_vectors = (predictions, given)

        return model, final_metrics, label_vectors, training_history

    def setup_and_train_network(self):
        # Data
        # # Loading
        train_data, train_data_pred, test_data, valid_data = self.load_data()

        # Model
        # # Definition
        model = self.define_model(train_data[0].shape, train_data[1].shape)

        # # Training
        training_history = self.train_model(model, train_data, valid_data)

        # # Calculate accuracy
        final_metrics = self.calc_categorical_accuracy(
            model, train_data_pred, test_data
        )

        # # Calulate prediction
        predictions, given = self.is_vs_should_categorical(model, test_data)
        label_vectors = (predictions, given)

        # keras_backend.clear_session()

        return model, final_metrics, label_vectors, training_history


if __name__ == "__main__":
    from tensorflow.python.client import device_lib

    logger.debug("ENVIRONMENT")
    for device in device_lib.list_local_devices():
        logger.debug(
            f"{device.name}:{device.device_type} with memory {device.memory_limit}"
        )
        logger.debug(f" {device.physical_device_desc}")

    hyperparameters = HyperParameters(model_name=model_name, loglevel=logging.DEBUG)
    neural_network = LSTMLayerN(hyperparameters)
    neural_network.setup_and_train_network()
    neural_network.train_network(100)
