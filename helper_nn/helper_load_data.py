import os
import numpy as np
from random import shuffle
import numpy as np
import pandas as pd
from typing import Union, List

# pathes
HOME_PATH = os.getcwd()
DATA_PATH = os.path.join("data")

#########################################################################################
#                                                                                       #
#                                    UCI HAR Dataset                                    #
# from helper_nn.helper_load_data import uci_har_dataset_data as data                   #
# from helper_nn.helper_load_data import INPUT_SIGNAL_TYPES,LABELS                      #
# #######################################################################################
# Labels of input columns
INPUT_SIGNAL_TYPES = [
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
LABELS = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]


def uci_har_dataset_data(
    colums_to_use: List[int]
) -> (np.array, np.array, np.array, np.array):
    """Numpy Array of input and labels for training and test data

    :param colums_to_use: Integer of columns to be used (corresponds to INPUT_SIGNAL_TYPES)
    :return: X_train, y_train, X_test, y_test
    """
    DATASET_PATH = os.path.join(DATA_PATH, "UCI_HAR_Dataset")
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
                for elem in [row.replace("  ", " ").strip().split(" ") for row in file]
            ],
            dtype=np.int32,
        )
        file.close()

        # Substract 1 to each output class for friendly 0-based indexing
        return y_ - 1

    # print("We use the features", [INPUT_SIGNAL_TYPES[i] for i in colums_to_use])

    X_train_signals_paths = [
        os.path.join(DATASET_PATH, TRAIN_DATA, "Inertial Signals", signal + "train.txt")
        for signal in INPUT_SIGNAL_TYPES
    ]
    X_test_signals_paths = [
        os.path.join(DATASET_PATH, TEST_DATA, "Inertial Signals", signal + "test.txt")
        for signal in INPUT_SIGNAL_TYPES
    ]
    X_train = load_X(X_train_signals_paths)[:, :, colums_to_use]
    X_test = load_X(X_test_signals_paths)[:, :, colums_to_use]

    y_train_path = os.path.join(DATASET_PATH, TRAIN_DATA, "y_train.txt")
    y_test_path = os.path.join(DATASET_PATH, TEST_DATA, "y_test.txt")
    y_train = load_y(y_train_path)
    y_test = load_y(y_test_path)

    idx_shuffel = np.arange(X_train.shape[0])
    shuffle(idx_shuffel)

    X_train = X_train[idx_shuffel, :, :]
    y_train = y_train[idx_shuffel]

    return X_train, y_train, X_test, y_test
