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
#                                    MNIST Dataset                                      #
# from helper.helper_load_data import mnist_data as data                             #
# from helper.helper_load_data import MNIST_DATASET_PATH as DATASET_PATH             #
# from helper.helper_load_data import MNIST_LABELS as LABELS                         #
# #######################################################################################
MNIST_DATASET_PATH = os.path.join(DATA_PATH, "MNIST")
MNIST_LABELS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
MNIST_DATASET = None

def mnist_data():
    # ToDo
    from tensorflow.examples.tutorials.mnist import input_data
    global MNIST_DATASET
    mnist = input_data.read_data_sets(MNIST_DATASET_PATH, one_hot=True)
    X_train, y_train = mnist.train.next_batch(mnist.train.num_examples)
    X_test, y_test = mnist.test.next_batch(mnist.test.num_examples)
    MNIST_DATASET = mnist
    return X_train, y_train, X_test, y_test

#########################################################################################
#                                                                                       #
#                           Boston Housing Dataset                                      #
# from helper.helper_load_data import boston_housing_data as data                    #
# from helper.helper_load_data import BOSTON_DATASET_PATH as DATASET_PATH            #
# from helper.helper_load_data import BOSTON_LABELS as LABELS                        #
# #######################################################################################
from sklearn.model_selection import train_test_split
from scipy.stats import zscore

BOSTON_DATASET_PATH = os.path.join(DATA_PATH, "BostonHousing")
BOSTON_LABELS = ["a", "b", "c", "d", "e", "f", "g"]


def boston_housing_data():
    # Read the data set
    df = pd.read_csv(
        "https://data.heatonresearch.com/data/t81-558/jh-simple-dataset.csv",
        na_values=["NA", "?"],
    )

    # Generate dummies for job
    df = pd.concat([df, pd.get_dummies(df["job"], prefix="job")], axis=1)
    df.drop("job", axis=1, inplace=True)

    # Generate dummies for area
    df = pd.concat([df, pd.get_dummies(df["area"], prefix="area")], axis=1)
    df.drop("area", axis=1, inplace=True)

    # Missing values for income
    med = df["income"].median()
    df["income"] = df["income"].fillna(med)

    # Standardize ranges
    df["income"] = zscore(df["income"])
    df["aspect"] = zscore(df["aspect"])
    df["save_rate"] = zscore(df["save_rate"])
    df["age"] = zscore(df["age"])
    df["subscriptions"] = zscore(df["subscriptions"])

    # Convert to numpy - Classification
    x_columns = df.columns.drop("product").drop("id")
    x = df[x_columns].values
    dummies = pd.get_dummies(df["product"])  # Classification
    products = dummies.columns
    y = dummies.values

    # Split into train/test
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42)

    return x_train, y_train, x_test,  y_test


#########################################################################################
#                                                                                       #
#                                    UCI HAR Dataset                                    #
# from helper.helper_load_data import uci_har_dataset_data as data                   #
# from helper.helper_load_data import UCI_HAR_INPUT_SIGNAL_TYPESas INPUT_SIGNAL_TYPES#
# from helper.helper_load_data import UCI_HAR_LABELS as LABELS                       #
# #######################################################################################
# Labels of input columns
UCI_HAR_INPUT_SIGNAL_TYPES = [
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
UCI_HAR_LABELS = [
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
        for signal in UCI_HAR_INPUT_SIGNAL_TYPES
    ]
    X_test_signals_paths = [
        os.path.join(DATASET_PATH, TEST_DATA, "Inertial Signals", signal + "test.txt")
        for signal in UCI_HAR_INPUT_SIGNAL_TYPES
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
