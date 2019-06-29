# This dataloader loads the "motionsense-dataset"
# http://archive.ics.uci.edu/ml/datasets/EEG+Eye+State
#
# Usage:
# from dataloader.eeg_eyes import Loader
# dataloader = Loader()
# trainX, trainY, trainX_forPred, trainY_forPred, valX, valY, testX, testY = dataloader.eeg_eyes(<batches>)
import os
import re
import glob
import numpy as np
import pandas as pd
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
        self.dataset_path = os.path.join(self.home_path, self.data_path, "EEG")
        self.filename = r"EEG Eye State.arff"
        # Labels of input columns
        self.input_signal_types = [
            "AF3",
            "F7",
            "F3",
            "FC5",
            "T7",
            "P7",
            "O1",
            "O2",
            "P8",
            "T8",
            "FC6",
            "F4",
            "F8",
            "AF4",
        ]
        # Output classes to learn how to classify
        self.label = "eyeDetection"
        self.classes = ["EyeClosed", "EyeOpen"]

        #
        ## train-test-splitt:
        ## define point in time t0 where we split the time series
        ## train is before t0
        ## test is after t0
        self.t0 = 12000

        ## ts_length
        ## timeseries 0-t0 itd aufgeteilt in
        ## kleinere timeserie der Länge ts_length
        ## alle ts_length/batches fängt eine neue kleine timeseries an
        ## d.h. 1.timeseries   0 - 2000 (0 - ts_length)
        ##      2.timeseries 500 - 2500 (1 * ts_length/batches - (1 * ts_length/batches +ts_length) )
        ##      ...
        self.ts_length = 2000

    def eeg_data(self, batches):
        """Numpy Array of input and labels for training and test data

        :param batches: Number of batches
        :return: trainX, trainY, trainX_forPred, trainY_forPred, valX, valY, testX, testY
        """

        def prep_time_data(start, end):
            X1 = np.expand_dims(X[start:end, :], axis=0)
            Y1 = y[start:end].reshape(-1, 1)
            Y1 = np.expand_dims(np.hstack([Y1, 1 - Y1]), axis=0)
            return X1, Y1

        assert (
            self.ts_length / batches == self.ts_length // batches
        ), "batches must divide ts_length"
        data = pd.read_csv(
            os.path.join(self.dataset_path, self.filename),
            names=self.input_signal_types + [self.label],
            skiprows=19,
            index_col=False,
        )

        # PREPARE DATA
        X = data[self.input_signal_types].values
        y = data[self.label].values

        scaler = preprocessing.RobustScaler()
        scaler.fit(X[: self.t0, :])
        X = scaler.transform(X)
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)
        ## cut signal at -1 and 1
        # X = X/2 ## alternative: cut at -2 and 2 and scale
        idx1 = X > 1
        idx2 = X < -1
        X_peaks = np.zeros_like(X)
        X[idx1] = 1
        X[idx2] = -1
        ## and add new columns which indicate if there was a peak
        X_peaks[idx1] = 1
        X_peaks[idx2] = -1
        X = np.hstack([X, X_peaks])

        ## divide dataset into step_size segments,
        #  each segment corrsesponds to one of the batches
        step_size = self.ts_length // batches  # 1000

        n_ts = self.t0 // self.ts_length - 1
        first_start = self.t0 - (n_ts + 1) * self.ts_length  # = 0
        ts_starts = np.array(
            range(first_start, self.t0 - step_size - self.ts_length + 1, step_size)
        )
        windows = pd.DataFrame(
            {
                "batch": list(range(batches)) * n_ts,
                "start": ts_starts,
                "end": ts_starts + self.ts_length,
            }
        )
        ## build training set
        X_list = []
        Y_list = []
        for start in range(
            first_start, self.t0 - step_size - self.ts_length + 1, step_size
        ):
            X1, Y1 = prep_time_data(start, start + self.ts_length)
            X_list.append(X1)
            Y_list.append(Y1)
        # put into 3D Tensor (number of small time series, lenght of small time series, inputs)
        trainX = np.vstack(X_list)
        trainY = np.vstack(Y_list)

        ## validation set to observe while training
        X_list = []
        Y_list = []
        for i in range(1 - batches, 1):
            start = self.t0 + i * step_size
            X1, Y1 = prep_time_data(start, start + self.ts_length)
            X_list.append(X1)
            Y_list.append(Y1)

        valX = np.vstack(X_list)
        valY = np.vstack(Y_list)

        ## test set
        n_pred = self.ts_length  # 2000
        trainX_forPred, trainY_forPred = prep_time_data(
            self.t0 - n_pred, self.t0
        )  ## used to generate valid state
        testX, testY = prep_time_data(self.t0, self.t0 + n_pred)

        return trainX, trainY, trainX_forPred, trainY_forPred, valX, valY, testX, testY
