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

import tensorflow.python.keras.api._v2.keras as keras

DEBUG = False


class Loader:
    def __init__(self):
        appendix = "_bis_aug19"
        self.home_path = os.getcwd()
        self.data_path = os.path.join("data")
        self.dataset_path = os.path.join(self.home_path, self.data_path, "XAU")
        self.filename = f"test_nn_XAUUSD{appendix}.h5"
        self.instrument = "XAUUSD"
        # Labels of input columns
        self.input_signal_types = ["XAU Timeseries"]
        # Output classes to learn how to classify
        self.label = "Long-Short"
        self.classes = ["Short", "Shortish", "Longish", "Long"] #["Short", "Shortish", "Flat", "Longish", "Long"]
        #           Short R        Long R
        # Short:      n              -1
        # Shortish:  0<x<n           <0
        # Flat:      <0              <0
        # Longish:   <0             0<x<n
        # Long:      -1               n
        self.epsilon = 0.0001

    def read_dataframe_from_file(self, path: str, name: str, mode="r") -> DataFrame:
        """Reads dataframe from hdf5 file

        :param path: Full path to hdf5 file
        :param name: Name of data to be read
        :param mode: open mode (read "r", readwrite "r+")

        :return: Read dataframe
        """
        df = pd.read_hdf(path, name, mode)
        assert isinstance(df, DataFrame)
        return df

    def szenario_name(self, rr, period, atr, long=True):
        if long:
            start = "Long"
        else:
            start = "Short"
        name = f"{start}_RR{rr:.0f}_P{period:.0f}_A{atr:.0f}"
        return name

    def szenario_numbers(self, name):
        rr, period, atr = [
            int("".join(filter(lambda x: x.isdigit(), number)))
            for number in name.split("_")[1:]
        ]
        atr = atr
        return (rr, period, atr)

    def load_data(self, rr, period, atr, look_back_period):
        df = self.read_dataframe_from_file(
            os.path.join(self.dataset_path, self.filename), self.instrument
        )
        index = df.index
        columns = df.columns
        price_input_cols = ["open", "high", "low", "close"]
        output_scenarios = [
            self.szenario_numbers(name) for name in df.columns if "Long_" in name
        ]
        if (rr, period, atr) in output_scenarios:
            # classify output (one hot)
            long_name = self.szenario_name(rr, period, atr, long=True)
            short_name = self.szenario_name(rr, period, atr, long=False)

            short_win = df[short_name] > (rr - self.epsilon)
            short_ok = (df[short_name] > 0) & ~short_win
            short_loss = df[short_name] < 0
            long_win = df[long_name] > (rr - self.epsilon)
            long_ok = (df[long_name] > 0) & ~long_win
            long_loss = df[long_name] < 0
            short = short_win
            short[short] = 1
            shortish = short_ok
            shortish[shortish] = 1
            flat = short_loss & long_loss
            #flat[flat] = 1
            longish = long_ok
            longish[longish] = 1
            long = long_win
            long[long] = 1
            #df_one_hot = pd.concat(
            #    [short, shortish, flat, longish, long], keys=self.classes, axis=1
            #)
            df_one_hot = pd.concat(
                [short, shortish,  longish, long], keys=self.classes, axis=1
            )
            # transform input
            df_prices = df[price_input_cols]
            min = df_prices.min().min() * 0.9
            max = df_prices.max().max() * 1.1
            df_prices = np.log10(df_prices / max) + 1.0
            df_input = df_prices

            # remove invalid columns
            nan = df[long_name].isna() | df[short_name].isna()
            df_input = df_input[~(nan|flat)]
            df_one_hot = df_one_hot[~(nan|flat)]

            # Transformed data
            df_all = pd.concat([df_input, df_one_hot], axis=1)

            # split into datasets of certain length
            split = (0.8, 0.2, 0)
            X_list = []
            Y_list = []
            for end in range(look_back_period, len(df_all) + 1, 1):
                temp_df = df_all.iloc[end - look_back_period : end]
                #if temp_df["Flat"].iloc[-1] == 1 : # and end < split[0]*len(df_all):
                #    continue
                X1, Y1 = temp_df[price_input_cols].values, temp_df[self.classes].iloc[-1].values
                X_list.append(X1)
                Y_list.append(Y1)
            number_of_sets = len(X_list)
            # split train & test data

            X_train = np.array(X_list[: int(number_of_sets * split[0])])
            y_train = np.array(Y_list[: int(number_of_sets * split[0])])
            X_validate = np.array(
                X_list[
                    int(number_of_sets * split[0]) : int(
                        number_of_sets * (split[0] + split[1])
                    )
                ]
            )

            y_validate = np.array(
                Y_list[
                    int(number_of_sets * split[0]) : int(
                        number_of_sets * (split[0] + split[1])
                    )
                ]
            )
            X_test = np.array(X_list[int(number_of_sets * (split[0] + split[1])) :])
            y_test = np.array(Y_list[int(number_of_sets * (split[0] + split[1])) :])
        else:
            X_train, y_train, X_validate, y_validate, X_test, y_test = (
                None,
                None,
                None,
                None,
                None,
                None,
            )
        return X_train, y_train, X_validate, y_validate, X_test, y_test

if __name__ == "__main__":
    loader = Loader()
    X_train, y_train, X_validate, y_validate, X_test, y_test  = loader.load_data(4, 20, 100, 20)
