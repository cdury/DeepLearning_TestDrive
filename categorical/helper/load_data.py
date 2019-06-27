# general imports
import os
import re
import glob
import itertools
from random import shuffle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as skm
from sklearn import preprocessing as skp

# typing imports
from typing import Union, List

# keras imports
from tensorflow.python.keras.api._v2 import keras

# pathes
HOME_PATH = os.getcwd()
DATA_PATH = os.path.join("data")
########################################################################################
#                                                                                       #
#     Motions sense                                                                     #
# from helper.helper_load_data import motion_data as data                                  #
# from helper.helper_load_data import MOTION_DATASET_PATH as DATASET_PATH                  #
# from helper.helper_load_data import MOTION_INPUT_SIGNAL_TYPES as INPUT_SIGNAL_TYPES      #
# from helper.helper_load_data import MOTION_LABELS as LABELS                              #
# #######################################################################################
# Labels of input columns
MOTION_DATASET_PATH = os.path.join(HOME_PATH,DATA_PATH, "motionsense-dataset")
MOTION_INPUT_SIGNAL_TYPES = [
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
MOTION_CATEGORIES= [
    ## categorial
    'subject', # label = 'subject' ## (1-24)
    'gender', # label = 'gender' ##  (F:0,M:1)
    'activity',# label = 'activity' ## is simple to predict -> try yourself
    ## numeric
    'weight' # label = 'weight'
]
MOTION_LABELS = None


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

def motion_dataframes(label,T):
    path = os.path.join(MOTION_DATASET_PATH, "A_DeviceMotion_data")

    data = MotionData(path).parse_files()

    data.dfs[0][MOTION_INPUT_SIGNAL_TYPES].plot(figsize=(15, 5))  # Plot Values of fisrt dataset
    plt.show()

    df_list = []
    k_list = []
    for df in data.dfs:
        t = df.shape[0]
        k = t // T
        k_list.append(k)
        for i in range(k):
            df_list.append(df.iloc[i * T: (i + 1) * T, :])

    ## load and join labels
    # %%
    sub_info = pd.read_csv(
        os.path.join(MOTION_DATASET_PATH, "data_subjects_info.csv")
    )

    print(sub_info.head())
    # %%
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
    print(label_df.head(20))

    # #########################
    ## check label distribution
    print(label_df[label].value_counts())
    ## Train-test-split
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
            stratify=label_df[label],  ## make sure the subjects are equally distributed
        )

    return df_list, label_df, idx_train, idx_test

def motion_data(label=None, n_timesteps=0, debug=False):

    df_list, label_df, idx_train, idx_test = motion_dataframes(label, n_timesteps)
    n_inputs = len(MOTION_INPUT_SIGNAL_TYPES)
    n_samples = len(df_list)
    # #########################
    # NNs
    ## encode label
    le = skp.OneHotEncoder(categories="auto")
    y = le.fit_transform(label_df[label].values.reshape(-1, 1))
    y_train = y[idx_train]
    y_test = y[idx_test]

    ## Preproc for LSTM and 1D-CNN
    ## fit scaler on all values
    scaler = skp.StandardScaler()
    all_values = pd.concat(df_list)
    scaler.fit(all_values[MOTION_INPUT_SIGNAL_TYPES].values)

    X = np.zeros((n_samples, n_timesteps, n_inputs))
    for i, df in enumerate(df_list):
        # print(i)
        val = df[MOTION_INPUT_SIGNAL_TYPES].values
        val = scaler.transform(val)
        X[i, :, :] = val
    X_train = X[idx_train]
    X_test = X[idx_test]

    return X_train, y_train,   X_test, y_test, df_list, label_df
#########################################################################################
#                                                                                       #
#     EEG Eyes Dataset    (http://archive.ics.uci.edu/ml/datasets/EEG+Eye+State)        #
# from helper.helper_load_data import eeg_data as data                                  #
# from helper.helper_load_data import EEG_DATASET_PATH as DATASET_PATH                  #
# from helper.helper_load_data import EEG_INPUT_SIGNAL_TYPES as INPUT_SIGNAL_TYPES      #
# from helper.helper_load_data import EEG_LABELS as LABELS                              #
# #######################################################################################
# Labels of input columns
EEG_INPUT_SIGNAL_TYPES = [
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
EEG_LABELS = ["EyeClosed", "EyeOpen"]


def eeg_data(batches):
    """Numpy Array of input and labels for training and test data

    :param batches: Number of batches
    :return: X_train, y_train, X_test, y_test
    """

    def prep_time_data(start, end):
        X1 = np.expand_dims(X[start:end, :], axis=0)
        Y1 = y[start:end].reshape(-1, 1)
        Y1 = np.expand_dims(np.hstack([Y1, 1 - Y1]), axis=0)
        return X1, Y1

    #
    ## train-test-splitt:
    ## define point in time t0 where we split the time series
    ## train is before t0
    ## test is after t0
    t0 = 12000
    ## ts_length
    ## timeseries 0-t0 itd aufgeteilt in
    ## kleinere timeserie der Länge ts_length
    ## alle ts_length/batches fängt eine neue kleine timeseries an
    ## d.h. 1.timeseries   0 - 2000 (0 - ts_length)
    ##      2.timeseries 500 - 2500 (1 * ts_length/batches - (1 * ts_length/batches +ts_length) )
    ##      ...
    ts_length = 2000
    assert ts_length / batches == ts_length // batches, "batches must divide ts_length"
    #
    DATASET_PATH = os.path.join(HOME_PATH,DATA_PATH, "EEG")
    file = r"EEG Eye State.arff"
    cols = [
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
        "eyeDetection",
    ]
    data = pd.read_csv(
        os.path.join(DATASET_PATH, file), names=cols, skiprows=19, index_col=False
    )

    # PREPARE DATA
    label = "eyeDetection"
    X = data[EEG_INPUT_SIGNAL_TYPES].values
    y = data[label].values

    scaler = skp.RobustScaler()
    scaler.fit(X[:t0, :])
    X = scaler.transform(X)
    le = skp.LabelEncoder()
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
    n_pred = ts_length  # 2000
    step_size = ts_length // batches  # 1000

    n_ts = t0 // ts_length - 1
    first_start = t0 - (n_ts + 1) * ts_length
    ts_starts = np.array(range(first_start, t0 - step_size - ts_length + 1, step_size))
    windows = pd.DataFrame(
        {
            "batch": list(range(batches)) * n_ts,
            "start": ts_starts,
            "end": ts_starts + ts_length,
        }
    )
    ## build training set
    X_list = []
    Y_list = []
    for start in range(first_start, t0 - step_size - ts_length + 1, step_size):
        X1, Y1 = prep_time_data(start, start + ts_length)
        X_list.append(X1)
        Y_list.append(Y1)
    # put into 3D Tensor (number of small time series, lenght of small time series, inputs)
    trainX = np.vstack(X_list)
    trainY = np.vstack(Y_list)

    ## validation set to observe while training
    X_list = []
    Y_list = []
    for i in range(1 - batches, 1):
        start = t0 + i * step_size
        X1, Y1 = prep_time_data(start, start + ts_length)
        X_list.append(X1)
        Y_list.append(Y1)

    valX = np.vstack(X_list)
    valY = np.vstack(Y_list)

    ## test set
    trainX_forPred, trainY_forPred = prep_time_data(
        t0 - n_pred, t0
    )  ## used to generate valid state
    testX, testY = prep_time_data(t0, t0 + n_pred)

    return trainX, trainY, trainX_forPred, trainY_forPred, valX, valY, testX, testY


#########################################################################################
#                                                                                       #
#                                    MNIST Dataset                                      #
# from helper.helper_load_data import mnist_data as data                                #
# from helper.helper_load_data import MNIST_DATASET_PATH as DATASET_PATH                #
# from helper.helper_load_data import MNIST_LABELS as LABELS                            #
# #######################################################################################
MNIST_DATASET_PATH = os.path.join(DATA_PATH, "MNIST")
MNIST_LABELS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


def mnist_data():
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
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
        x, y, test_size=0.25, random_state=42
    )

    return x_train, y_train, x_test, y_test


#########################################################################################
#                                                                                       #
#                                    UCI HAR Dataset                                    #
# from helper.helper_load_data import uci_har_dataset_data as data                      #
# from helper.helper_load_data import UCI_HAR_INPUT_SIGNAL_TYPES as INPUT_SIGNAL_TYPES  #
# from helper.helper_load_data import UCI_HAR_LABELS as LABELS                          #
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
