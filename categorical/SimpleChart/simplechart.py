# This dataloader loads the "Simple Chart Data"
# Folder: ???
#
# Usage:
# from simplechart import Loader
# dataloader = Loader()
# x_train, y_train, x_test, y_test = dataloader.boston_housing_data()
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
        self.dataset_path = os.path.join(self.home_path, self.data_path, "MNIST")
        self.filename = r""
        # Labels of input columns
        self.input_signal_types = ["28x28 Pictures"]
        # Output classes to learn how to classify
        self.label = "Numbers"
        self.classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    def load_data(self):
        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
        return X_train, y_train, X_test, y_test
