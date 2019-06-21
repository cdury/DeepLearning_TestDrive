# general imports
import os, sys
import time
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

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
# # Model
Model = keras.models.Model
# # Regularizers
l2 = keras.regularizers.l2
# # Optimizer
Adam = keras.optimizers.Adam
# # Utils
plot_model = keras.utils.plot_model

model_name = os.path.splitext(os.path.basename(__file__))[0]

# helper imports
from helper.helper_encoding import one_hot

##############################################################################################
# Parameters
##############################################################################################
# Data
from helper.helper_load_data import boston_housing_data as data
from helper.helper_load_data import BOSTON_DATASET_PATH as DATASET_PATH
from helper.helper_load_data import BOSTON_LABELS as LABELS

# Network
from networks.Base_Supervised_Categorical import (
    BaseParameters,
    BaseNN,
    timing,
)

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
        self.labels = LABELS  # Labels of th categorizations

        # # Modell (Hyperparamters)
        self.n_hidden_1 = 100  # Num of features in the first hidden layer
        self.n_hidden_2 = 50  # Num of features in the first hidden layer
        self.n_hidden_3 = 25  # Num of features in the first hidden layer
        self.n_hidden_4 = 1  # Num of features in the first hidden layer

        self.init_kernel = "random_normal"  # "he_normal", 'random_normal'
        self.activation_hidden = "relu"
        self.activation_output = "softmax"

        # # Optimizer (Hyperparamters)
        self.learning_rate = 0.001
        self.lambda_loss_amount = 0.000191
        self.metric = "accuracy"

        # END Hyperparameter


##############################################################################################
# Neural Network
# API
##############################################################################################


class SNNLayerN(BaseNN):
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
    ]:
        # Import Boston data
        x_train, y_train, x_test, y_test = data()

        # # Features
        #y_train = one_hot(y_train)
        #y_test = one_hot(y_test)
        train_data = x_train, y_train
        test_data = x_test, y_test
        valid_data = np.ndarray([]), np.ndarray([])
        self.test_data = test_data
        self.validation_data = valid_data
        return train_data, test_data, valid_data

    @timing
    def define_model(self, input_shape, output_shape) -> Model:
        # Input (number of inputs)
        n_input = input_shape[1]
        self.parameter.n_input = n_input
        # Output (number of classes)
        n_classes = output_shape[1]
        self.parameter.n_classes = n_classes

        # Start defining the input tensor:
        input_layer = Input((n_input,))

        # create the layers and pass them the input tensor to get the output tensor:
        layer_1 = Dense(
            units=self.parameter.n_hidden_1,
            activation=self.parameter.activation_hidden,
            #kernel_regularizer=l2(self.parameter.lambda_loss_amount),
            kernel_initializer=self.parameter.init_kernel,
        )(input_layer)

        layer_2 = Dense(
            units=self.parameter.n_hidden_2,
            activation=self.parameter.activation_hidden,
            #kernel_regularizer=l2(self.parameter.lambda_loss_amount),
            kernel_initializer=self.parameter.init_kernel,
        )(layer_1)

        layer_3 = Dense(
            units=self.parameter.n_hidden_3,
            activation=self.parameter.activation_hidden,
            #kernel_regularizer=l2(self.parameter.lambda_loss_amount),
            kernel_initializer=self.parameter.init_kernel,
        )(layer_2)

        out_layer = Dense(
            units=self.parameter.n_classes,
            activation=self.parameter.activation_output,
            #kernel_initializer=self.parameter.init_kernel,
            # kernel_regularizer= l2(self.parameter.lambda_loss_amount),
        )(layer_3)

        # Define the model's start and end points
        model = Model(inputs=input_layer, outputs=out_layer)

        # Define the loss function
        loss_fn = lambda y_true, y_pred: tf.nn.softmax_cross_entropy_with_logits(
            logits=y_pred, labels=y_true
        )
        # loss_fn = "categorical_crossentropy"

        # Define the optimizer
        optimizer_fn = Adam(lr=self.parameter.learning_rate)
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

        return model

    @timing
    def train_model(self, model, train_data, validation_data):
        # Data
        x_train, y_train = train_data
        x_test, y_test = validation_data

        # Callbacks
        tensorboard, checkpoint, earlyterm = self.get_callbacks()
        used_callbacks = [tensorboard, earlyterm]

        # Training
        if self.parameter.batch_size > 0:
            history = model.fit(
                x_train,
                y_train,
                validation_data=(x_test, y_test),
                batch_size=self.parameter.batch_size,
                epochs=self.parameter.epochs,
                verbose=self.parameter.fitting_verbosity,
                callbacks=used_callbacks,
            )
        else:
            history = model.fit(
                x_train,
                y_train,
                validation_data=(x_test, y_test),
                epochs=self.parameter.epochs,
                verbose=self.parameter.fitting_verbosity,
                callbacks=used_callbacks,
            )
        logger.info("Training Finished!")
        return history

    def setup_and_train_network(self):
        # Data
        # # Loading
        train_data, test_data, _ = self.load_data()
        # EITHER
        # with keras_backend.get_session() as sess: # get active tf-session
        # OR
        # config = tf.compat.v1.ConfigProto(device_count={"GPU": 1, "CPU": 1})
        # config.gpu_options.allow_growth = True
        # with tf.compat.v1.Session(config=config) as sess:
        #     keras_backend.set_session(sess)
        #     # END EITHER

        # Model
        # # Definition
        model = self.define_model(train_data[0].shape, train_data[1].shape)

        # # Training
        training_history = self.train_model(model, train_data, test_data)

        # # Calculate accuracy
        final_metrics = self.calc_categorical_accuracy(model, test_data)

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
    neural_network = SNNLayerN(hyperparameters)
    neural_network.setup_and_train_network()
