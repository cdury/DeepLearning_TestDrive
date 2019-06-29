# ToDo This dataloader loads the ".... dataset"
# ToDo http://???
#
# Usage:
# ToDo from ... import Loader
# dataloader = Loader()
# ToDo X_train, y_train, X_valid, y_valid, X_test, y_test = dataloader...._data()
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

import tensorflow.python.keras.api._v2.keras as keras

DEBUG = False


class Loader:
    def __init__(self):
        self.home_path = os.getcwd()
        self.data_path = os.path.join("data")
        self.dataset_path = os.path.join(self.home_path, self.data_path, "...")  # ToDo
        self.filename = r"..."  # ToDo
        # Labels of input columns
        self.input_signal_types = ["..."]  # ToDo: Name of all input signals/channels
        self.labels = ["..."]  # ToDo: List of all possible labels
        # Output classes to learn how to classify
        self.label_classes = {}
        self.label_classes["..."] = ["..."]  # ToDo: All Classes of the labels

    def _data(self):
        # ToDo: Code for gathering/compiling the data suitable for the network
        # ToDo: Loading data
        # ToDo: Process data if neccesary
        # ToDo: Normalize data (zscore, onehot, ... ) if necessary
        (X_train, y_train) = np.ndarray([]), np.ndarray([])  # ToDo
        (X_valid, y_valid) = np.ndarray([]), np.ndarray([])  # ToDo
        (X_test, y_test) = np.ndarray([]), np.ndarray([])  # ToDo
        return X_train, y_train, X_valid, y_valid, X_test, y_test
