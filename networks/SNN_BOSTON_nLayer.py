# general imports
import os, sys
import time
import numpy as np
import pandas as pd
import tensorflow as tf

# typing imports
from typing import Tuple, List, Any, Union, Optional
from numpy import ndarray
from pandas import DataFrame, Series

# keras imports
from tensorflow.contrib.keras.api.keras.layers import Input, Dense
from tensorflow.contrib.keras.api.keras.models import Model
from tensorflow.contrib.keras.api.keras.optimizers import Adam
from tensorflow.contrib.keras.api.keras.callbacks import (
    TensorBoard,
    ModelCheckpoint,
    EarlyStopping,
)
import tensorflow.contrib.keras.api.keras.backend as keras_backend

model_name = "SNN_3_Layer"

# helper imports


##############################################################################################
# Parameters
##############################################################################################
# Data
from helper_nn.helper_load_data import boston_housing_data as data
from helper_nn.helper_load_data import BOSTON_DATASET_PATH as DATASET_PATH
from helper_nn.helper_load_data import BOSTON_LABELS as LABELS

# Network
from networks.Base_Supervised_Categorical import BaseParameters, BaseNN, timing


class HyperParameters(BaseParameters):
    """ HyperParamters for the neural network

    """

    def __init__(self, model_name, loglevel):
        # Process parameter
        super().__init__(model_name, loglevel)
        self.model_name = model_name

        # Hyperparameter
        # # Data
        self.colums_to_use = [0, 1]  # List of data columns to be used

        # # Modell
        self.n_input = None # Set in code
        self.n_hidden_1 = 100  # Num of features in the first hidden layer
        self.n_hidden_2 = 50  # Num of features in the first hidden layer
        self.n_hidden_3 = 25  # Num of features in the first hidden layer
        self.n_hidden_4 = 1  # Num of features in the first hidden layer
        self.n_classes = None  # Number of classification classes (set in code)
        self.labels = LABELS
        self.init_kernel = "random_normal"  # "he_normal", 'random_normal'
        self.activation_hidden = "relu"
        self.activation_output = "softmax"
        # # Optimizer
        self.learning_rate = 0.001
        self.lambda_loss_amount = 0.000191
        self.metric = "accuracy"

        # # Training
        self.monitor = "val_loss"  # "val_acc"
        self.mode = "auto" #""min"  # "max"
        self.patience = 5
        self.batch_size = 0
        self.epochs = 1000

        # END Hyperparameter

        # Display
        self.display_iter = 30000
        # self.loglevel 0: All Msg 1: No INFO 2: No INFO & WARNING 3: No INFO, WARNING & ERROR
        self.callback_verbosity = 1  # 0: quiet // 1: update_messagse
        self.eval_verbosity = 0  # 0: quiet // 1: update_messagse
        self.fitting_verbosity = 2  # 0: quiet // 1: animation // 2: summary
        # # Checkpoint TensorBoard
        self.tb_update_freq = "epoch"
        self.tb_write_graph = True
        self.tb_write_images = True
        # # Checkpoint ModelCheckpoint
        self.save_every_epochs = 1


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
        train_data = x_train, y_train
        test_data = x_test, y_test
        valid_data = np.ndarray([]), np.ndarray([])
        self.test_data = test_data
        self.validation_data = valid_data
        return train_data, test_data, valid_data

    @timing
    def define_model(self, input_shape, output_shape) -> Model:
        # Input (number of inputs)
        dim = len(input_shape)
        if dim > 2:
            # ToDo: flatten
            pass
        n_input = input_shape[1]
        self.parameter.n_input = n_input
        # Output (number of classes)
        dim = len(output_shape)
        if dim > 2:
            # ToDo: flatten
            pass
        n_classes =output_shape[1]
        self.parameter.n_classes = n_classes

        # Start defining the input tensor:
        input_layer = Input((n_input,))

        # create the layers and pass them the input tensor to get the output tensor:
        layer_1 = Dense(
            units=self.parameter.n_hidden_1,
            activation=self.parameter.activation_hidden,
            # kernel_regularizer=regularizers.l2(0.0002),
            kernel_initializer=self.parameter.init_kernel,
        )(input_layer)

        layer_2 = Dense(
            units=self.parameter.n_hidden_2,
            activation=self.parameter.activation_hidden,
            # kernel_regularizer=regularizers.l2(0.0002),
            kernel_initializer=self.parameter.init_kernel,
        )(layer_1)

        layer_3 = Dense(
            units=self.parameter.n_hidden_3,
            activation=self.parameter.activation_hidden,
            # kernel_regularizer=regularizers.l2(0.0002),
            kernel_initializer=self.parameter.init_kernel,
        )(layer_2)

        out_layer = Dense(
            units=self.parameter.n_classes,
            activation=self.parameter.activation_output,
            kernel_initializer=self.parameter.init_kernel,
            # kernel_regularizer=regularizers.l2(0.0002),

        )(layer_3)

        # define the model's start and end points
        model = Model(inputs=input_layer, outputs=out_layer)

        # define the loss function
        loss_fn = lambda y_true, y_pred: tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=y_pred, labels=y_true
        )

        # put all components together
        model.compile(
            loss=loss_fn, # loss="categorical_crossentropy",
            optimizer=Adam(), #Adam(lr=learning_rate),
            #optimizer=tf.train.AdamOptimizer(
            #    learning_rate=self.parameter.learning_rate
            #),
            metrics=[self.parameter.metric],
        )
        if self.parameter.loglevel == 0:
            model.summary()

        return model

    @timing
    def train_model(self, model, train_data, validation_data):
        # Data
        x_train, y_train = train_data
        x_test, y_test = validation_data

        # Callbacks
        # for Tensorboard evaluation
        tensorboard = TensorBoard(
            log_dir=self.parameter.tensorboard_dir,
            update_freq=self.parameter.tb_update_freq,
            write_graph=self.parameter.tb_write_graph,
            write_images=self.parameter.tb_write_images,
            histogram_freq=0,
            write_grads=False,
        )
        # for saving network with weights
        checkpoint = ModelCheckpoint(
            filepath=os.path.join(
                self.parameter.model_dir, "weights.{epoch:03d}-{val_loss:.2f}.hdf5"
            ),
            monitor=self.parameter.monitor,
            save_best_only=True,
            mode=self.parameter.mode,
            verbose=self.parameter.callback_verbosity,
            period=self.parameter.save_every_epochs,
        )
        # for early termination
        earlyterm = EarlyStopping(
            monitor=self.parameter.monitor,
            mode=self.parameter.mode,
            patience=self.parameter.patience,
            restore_best_weights=True,
            verbose=self.parameter.callback_verbosity,
            min_delta=0.001,
        )
        callbacks = [tensorboard,  earlyterm]

        # Training
        if self.parameter.batch_size > 0:
            history = model.fit(
                x_train,
                y_train,
                validation_data=(x_test, y_test),
                batch_size=self.parameter.batch_size,
                epochs=self.parameter.epochs,
                verbose=self.parameter.fitting_verbosity,
                callbacks=callbacks,
            )
        else:
            history = model.fit(
                x_train,
                y_train,
                validation_data=(x_test, y_test),
                epochs=self.parameter.epochs,
                verbose=self.parameter.fitting_verbosity,
                callbacks=callbacks,
            )
        print("Training Finished!")
        return history

    def setup_and_train_network(self):
        # Data
        # # Loading
        train_data, test_data, _ = self.load_data()
        # EITHER
        # with keras_backend.get_session() as sess: # get active tf-session
        # OR
        config = tf.ConfigProto(device_count={"GPU": 1, "CPU": 1})
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            keras_backend.set_session(sess)
            # END EITHER

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

        keras_backend.clear_session()

        return model, final_metrics, label_vectors, training_history


if __name__ == "__main__":
    from tensorflow.python.client import device_lib

    print("ENVIRONMENT")
    print(device_lib.list_local_devices())
    print("")
    hyperparameters = HyperParameters(model_name=model_name, loglevel=0)
    neural_network = SNNLayerN(hyperparameters)
    neural_network.setup_and_train_network()
