# This dataloader loads the "UCI HAR Dataset   "
# http://???
#
# Usage:
# from dataloader.uci_har import Loader
# dataloader = Loader()
# trainX, trainY, trainX_forPred, trainY_forPred, valX, valY, testX, testY = dataloader.eeg_eyes(<batches>)
import os
import re
import glob
from random import shuffle
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


# Only for debugging
import matplotlib.pyplot as plt

# typing imports
from typing import Tuple, List, Any, Union, Optional
from numpy import ndarray
from pandas import DataFrame, Series
from scipy.sparse import csr_matrix

DEBUG = False


class Loader:
    def __init__(self):
        self.home_path = os.getcwd()
        self.data_path = os.path.join("data")
        self.dataset_path = os.path.join(
            self.home_path, self.data_path, "UCI_HAR_Dataset"
        )
        self.filename = r""
        # Labels of input columns
        self.input_signal_types = [
            "body_acc_x_",
            "body_acc_y_",
            "body_acc_z_",
            "body_gyro_x_",
            "body_gyro_y_",
            "body_gyro_z_",
            "total_acc_x_",
            "total_acc_y_",
            "total_acc_z_",
        ]
        # Output classes to learn how to classify
        self.label = "activityDetection"
        self.classes = [
            "WALKING",
            "WALKING_UPSTAIRS",
            "WALKING_DOWNSTAIRS",
            "SITTING",
            "STANDING",
            "LAYING",
        ]

    def uci_har_dataset_data(
        self, colums_to_use: List[int]
    ) -> (np.array, np.array, np.array, np.array):
        """Numpy Array of input and labels for training and test data

        :param colums_to_use: Integer of columns to be used (corresponds to INPUT_SIGNAL_TYPES)
        :return: X_train, y_train, X_test, y_test
        """
        TRAIN_DATA = "train/"
        TEST_DATA = "test/"

        # Load "X" (the neural network's training and testing inputs)
        def load_X(X_signals_paths):
            X_signals = []

            for signal_type_path in X_signals_paths:
                file = open(signal_type_path, "r")
                # Read dataset from disk, dealing with text files' syntax
                X_signals.append(
                    [
                        np.array(serie, dtype=np.float32)
                        for serie in [
                            row.replace("  ", " ").strip().split(" ") for row in file
                        ]
                    ]
                )
                file.close()

            return np.transpose(np.array(X_signals), (1, 2, 0))

        # Load "y" (the neural network's training and testing outputs)
        def load_y(y_path):
            file = open(y_path, "r")
            # Read dataset from disk, dealing with text file's syntax
            y_ = np.array(
                [
                    elem
                    for elem in [
                        row.replace("  ", " ").strip().split(" ") for row in file
                    ]
                ],
                dtype=np.int32,
            )
            file.close()

            # Substract 1 to each output class for friendly 0-based indexing
            return y_ - 1

        # print("We use the features", [self.input_signal_types[i] for i in colums_to_use])

        X_train_signals_paths = [
            os.path.join(
                self.dataset_path, TRAIN_DATA, "Inertial Signals", signal + "train.txt"
            )
            for signal in self.input_signal_types
        ]
        X_test_signals_paths = [
            os.path.join(
                self.dataset_path, TEST_DATA, "Inertial Signals", signal + "test.txt"
            )
            for signal in self.input_signal_types
        ]
        X_train = load_X(X_train_signals_paths)[:, :, colums_to_use]
        X_test = load_X(X_test_signals_paths)[:, :, colums_to_use]

        y_train_path = os.path.join(self.dataset_path, TRAIN_DATA, "y_train.txt")
        y_test_path = os.path.join(self.dataset_path, TEST_DATA, "y_test.txt")
        y_train = load_y(y_train_path)
        y_test = load_y(y_test_path)

        idx_shuffel = np.arange(X_train.shape[0])
        shuffle(idx_shuffel)

        X_train = X_train[idx_shuffel, :, :]
        y_train = y_train[idx_shuffel]

        return X_train, y_train, X_test, y_test
