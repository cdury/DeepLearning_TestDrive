# This dataloader loads the "motionsense-dataset"
# https://www.kaggle.com/malekzadeh/motionsense-dataset/home
# and/or
# https://github.com/mmalekzadeh/motion-sense
#
# Usage:
# from dataloader.motionsense import Loader
# dataloader = Loader()
# df_list, label_df, idx_train, idx_test = dataloader.motion_dataframes(<label>,<n_timesteps>)
#                                         - or -
# X_train, y_train, X_test, y_test, df_list, label_df = dataloader.motion_data(<label>,<n_timesteps>)
#
import os
import re
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Only for debugging
import matplotlib.pyplot as plt

# typing imports
from typing import Tuple, List, Any, Union, Optional
from numpy import ndarray
from pandas import DataFrame, Series
from scipy.sparse import csr_matrix

DEBUG = False


class MotionData:
    def __init__(self, path, file_ext="*.csv"):
        self.path = path
        self.file_ext = file_ext

    def parse_file(self, fn):
        df = pd.read_csv(fn)
        activity = os.path.basename(os.path.dirname(fn)).split("_")[0]
        subject = os.path.basename(fn)
        subject = int(re.findall("[0-9]+", subject)[0])

        self.files.append(fn)
        self.activity.append(activity)
        self.subject.append(subject)
        self.dfs.append(df)

    def parse_files(self):
        self.files = []
        self.dfs = []
        self.activity = []
        self.subject = []

        for act_path in glob.glob(os.path.join(self.path, "*")):
            for fn in glob.glob(os.path.join(act_path, self.file_ext)):
                self.parse_file(fn)

        self.activity = np.array(self.activity)
        self.subject = np.array(self.subject)

        return self


class Loader:
    def __init__(self):
        self.home_path = os.getcwd()
        self.data_path = os.path.join("data")
        self.dataset_path = os.path.join(
            self.home_path, self.data_path, "motionsense-dataset"
        )
        self.input_signal_types = [
            "attitude.roll",
            "attitude.pitch",
            "attitude.yaw",
            "gravity.x",
            "gravity.y",
            "gravity.z",
            "rotationRate.x",
            "rotationRate.y",
            "rotationRate.z",
            "userAcceleration.x",
            "userAcceleration.y",
            "userAcceleration.z",
        ]
        self.labels = [
            # # categorial
            "subject",  # label = 'subject' ## (1-24)
            "gender",  # label = 'gender' ##  (F:0,M:1)
            "activity",  # label = 'activity' ## is simple to predict -> try yourself
            # # numeric
            "weight",  # label = 'weight'
        ]
        self.classes = {}
        self.classes["subject"] = list(range(1, 25, 1))
        self.classes["gender"] = ["female", "male"]
        self.classes["activity"] = ["dws", "jog", "sit", "std", "ups", "wlk"]
        self.classes["weight"] = None

    def motion_dataframes(
        self, label: str, n_timesteps: int
    ) -> Tuple[List[DataFrame], DataFrame, ndarray, ndarray]:
        """ Returns label data for label partitioned into n_timestep junks. Also the indizes of the train & test-set.

        :param label: Name of the label to be the evaluted
        :param n_timesteps: Number of timestaps per sample
        :return:    df_list: List[DataFrame] df_list.shape(anzahl samples) DataFrame.shape(n_timesteps,idx+n_inputs)
                    label_df: DataFrame DataFrame.shape(anzahl samples,#label_classes + source_file_id)
                    idx_train, idx_test: indices of train/test-set
        """
        path = os.path.join(self.dataset_path, "A_DeviceMotion_data")

        data = MotionData(path).parse_files()

        if DEBUG:
            data.dfs[0][self.input_signal_types].plot(
                figsize=(15, 5)
            )  # Plot Values of first dataset
            plt.show()

        df_list = []
        k_list = []
        for df in data.dfs:
            t = df.shape[0]
            k = t // n_timesteps
            k_list.append(k)
            for i in range(k):
                df_list.append(df.iloc[i * n_timesteps : (i + 1) * n_timesteps, :])

        # # load and join labels
        sub_info = pd.read_csv(
            os.path.join(self.dataset_path, "data_subjects_info.csv")
        )
        if DEBUG:
            print(sub_info.head())
        label_df = pd.DataFrame(
            {
                "activity": data.activity.repeat(k_list),
                "subject": data.subject.repeat(k_list),
                "raw_df_nr": np.arange(len(data.dfs)).repeat(k_list),
            }
        )
        label_df = label_df.merge(sub_info, left_on="subject", right_on="code")
        label_df = label_df.sort_values("raw_df_nr")
        label_df.drop("code", axis=1, inplace=True)
        label_df.reset_index(drop=True, inplace=True)
        if DEBUG:
            print(label_df.head(20))

        # # ## ## ## ## ## ## ## ## ## ## ## ##
        # # check label distribution
        if DEBUG:
            print(label_df[label].value_counts())

        # # Train-test-split
        n = len(label_df)
        if label != "subject":
            idx_train, idx_test = train_test_split(
                np.array(range(n)), test_size=0.25, random_state=1
            )
        else:
            idx_train, idx_test = train_test_split(
                np.array(range(n)),
                test_size=0.25,
                random_state=1,
                stratify=label_df[
                    label
                ],  # # make sure the subjects are equally distributed
            )

        return df_list, label_df, idx_train, idx_test

    def motion_data(
        self, label: str = None, n_timesteps: int = 0
    ) -> Tuple[ndarray, csr_matrix, ndarray, csr_matrix, List[DataFrame], DataFrame]:
        """Returns x & y of train- & test-set; also the raw dataframes (data & label)

        :param label: Name of the label to be the evaluted
        :param n_timesteps: Number of timestaps per sample
        :return: x_train/x_test : ndarray.shape(n_samples-train/test , n_timesteps, n_inputs)
                 y_train/y_test : csr_matrix.shape(n_samples-train/test , n_classes)
                 df_list: List[DataFrame] df_list.shape(anzahl samples) DataFrame.shape(n_timesteps,idx+n_inputs)
                 label_df: DataFrame DataFrame.shape(anzahl samples,#label_classes + source_file_id)
        """

        df_list, label_df, idx_train, idx_test = self.motion_dataframes(
            label, n_timesteps
        )
        n_inputs = len(self.input_signal_types)
        n_samples = len(df_list)

        # # ## ## ## ## ## ## ## ## ## ## ## ##
        # NNs
        # # one-hot encode label (into  sparse matrix)
        le = OneHotEncoder(categories="auto")
        y = le.fit_transform(label_df[label].values.reshape(-1, 1))

        # # one-hot encode label (into ndarray)
        # from helper.helper_encoding import one_hot
        # y = label_df[label].values #.reshape(-1, 1)
        # y=one_hot(y,len(self.classes[label]))

        y_train = y[idx_train]
        y_test = y[idx_test]

        # # Preproc fit scaler on all values
        scaler = StandardScaler()
        all_values = pd.concat(df_list)
        scaler.fit(all_values[self.input_signal_types].values)

        X = np.zeros((n_samples, n_timesteps, n_inputs))
        for i, df in enumerate(df_list):
            val = df[self.input_signal_types].values
            val = scaler.transform(val)
            X[i, :, :] = val
        x_train = X[idx_train]
        x_test = X[idx_test]

        return x_train, y_train, x_test, y_test, df_list, label_df
