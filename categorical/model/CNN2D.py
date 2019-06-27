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
from tensorflow.python.keras.api._v1 import keras as kerasV1

# # Backend
keras_backend = keras.backend
# # Layers
Input = keras.layers.Input
Dense = keras.layers.Dense
LSTM = keras.layers.LSTM
GRU = keras.layers.GRU
CuDNNGRU = keras.layers.GRU  # kerasV1.layers.CuDNNGRU
GaussianNoise = keras.layers.GaussianNoise
Conv1D = keras.layers.Conv1D
Convolution1D = keras.layers.Convolution1D
MaxPooling1D = keras.layers.MaxPooling1D
MaxPooling2D = keras.layers.MaxPooling2D
GlobalAveragePooling1D = keras.layers.GlobalAveragePooling1D
Convolution2D = keras.layers.Convolution2D
TimeDistributed = keras.layers.TimeDistributed
Dropout  = keras.layers.Dropout
Flatten  = keras.layers.Flatten
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
from categorical.helper.encoding import one_hot

##############################################################################################
# Parameters
##############################################################################################

# Network
from categorical.model.Base_Supervised import BaseParameters, BaseNN, timing

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])


class NNParameters(BaseParameters):
    """ HyperParamters for the neural network

    """

    def __init__(self, model_name, dir_name, loglevel, parent_name=None):
        # Process parameter
        super().__init__(model_name, dir_name, loglevel, parent_name)
        # Filesystem & Paths
        # # data
        self.data_dir = self.data_path

        # Hyperparameter
        # # Data
        self.colum_names = None
        self.labels = None  # Labels of th categorizations
        # # Modell (Hyperparamters)
        self.n_conv_1 = 32
        self.n_conv_1_kernel = (3,3)
        self.n_conv_1_pooling = (2,2)
        self.n_hidden_2 = 32
        self.dropout = 0.1
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


class NNDefinition(BaseNN):
    def __init__(self, hyperparameter):
        super().__init__(hyperparameter)
        self.parameter: NNParameters = hyperparameter

    @timing
    def define_model(self, input_shape, output_shape) -> Model:
        # Input (number of inputs)
        dim_pic_x = input_shape[1]
        dim_pic_y = input_shape[2]
        n_input = input_shape[3]
        self.parameter.input_shape = (dim_pic_x, dim_pic_y, n_input)
        # Output (number of classes)
        n_classes = output_shape[1]
        self.parameter.n_classes = n_classes

        # Start defining the input tensor:
        model = Sequential()
        model.add(
            Convolution2D(
                filters=self.parameter.n_conv_1,
                kernel_size=self.parameter.n_conv_1_kernel,
                activation=self.parameter.activation_hidden,
                kernel_initializer=self.parameter.init_kernel,
                input_shape=(dim_pic_x, dim_pic_y, n_input),
                data_format="channels_last",
            )
        )

        model.add(MaxPooling2D(pool_size=self.parameter.n_conv_1_pooling))
        model.add(Dropout(self.parameter.dropout))
        # model.add(GaussianNoise(0.3))
        # model.add(Convolution2D(32, (3, 3),activation='relu'))
        # model.add(MaxPooling2D(pool_size=(2,2)))
        # model.add(Dropout(0.1))
        # model.add(GaussianNoise(0.1))
        model.add(Flatten())
        model.add(
            Dense(
                self.parameter.n_hidden_2,
                activation=self.parameter.activation_hidden,
                kernel_initializer=self.parameter.init_kernel,
            )
        )
        # model.add(LeakyReLU(alpha=0.03))
        model.add(Dropout(self.parameter.dropout))
        # model.add(GaussianNoise(0.2))
        model.add(
            Dense(
                self.parameter.n_classes,
                activation=self.parameter.activation_output,
                kernel_initializer=self.parameter.init_kernel,
            )
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
    def train_model(self, model, train_data, validation_data, initial_epoch=0):
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
        self.parameter.trained_epochs = history.epoch[-1] + 1
        logger.info("Training Finished!")
        return history

    def train_network(self, epochs=0):
        initial_epoch = self.parameter.trained_epochs
        train_data = self.train_data
        valid_data = self.validation_data
        test_data = self.test_data
        model = self.model
        self.parameter.epochs = epochs
        # # Training
        training_history = self.train_model(
            model, train_data, valid_data, initial_epoch=initial_epoch
        )

        # # Calculate accuracy
        final_metrics = self.calc_categorical_accuracy(model, valid_data)

        # # Calulate prediction
        predictions, given = self.is_vs_should_categorical(model, valid_data)
        label_vectors = (predictions, given)

        return model, final_metrics, label_vectors, training_history
