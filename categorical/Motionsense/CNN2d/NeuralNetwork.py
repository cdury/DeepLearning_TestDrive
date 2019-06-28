# general imports
import os
import copy
import logging
from categorical.helper.profiling import timing
import numpy as np
import pandas as pd
import sklearn.metrics as skm
from sklearn import preprocessing
import librosa
import matplotlib.pyplot as plt

# typing imports
from typing import Tuple, Union
from numpy import ndarray
from pandas import DataFrame, Series
from scipy.sparse import csr_matrix

# keras imports
from tensorflow.python.keras.api._v2 import keras

# # Backend
keras_backend = keras.backend
# # Layers
Input = keras.layers.Input
Dense = keras.layers.Dense
LSTM = keras.layers.LSTM
GaussianNoise = keras.layers.GaussianNoise
# # Model
Model = keras.models.Model
Sequential = keras.models.Sequential
# # Regularizers
l2 = keras.regularizers.l2
# # Optimizer
Adam = keras.optimizers.Adam
Adadelta = keras.optimizers.Adadelta
# # Utils
plot_model = keras.utils.plot_model

dir_name = os.path.split(os.path.split(os.path.dirname(__file__))[0])[1]
sub_dir_name = model_name = os.path.split(os.path.dirname(__file__))[1]
model_name = dir_name + "_" + sub_dir_name
dir_path = os.path.join(dir_name,sub_dir_name)

# helper imports

# Network
import categorical.model.CNN2D as CNN2D

# Data
from categorical.Motionsense.motionsense import Loader

logger = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])

##############################################################################################
# Parameters
##############################################################################################


class HyperParameters(CNN2D.NNParameters):
    """ HyperParamters for the neural network

    """

    def __init__(self, model_name, loglevel, parent_name=None):
        # Process parameter
        super().__init__(model_name, dir_name, loglevel, parent_name)
        # Loader
        loader = Loader()
        # # Data
        self.colum_names = copy.deepcopy(loader.input_signal_types)
        self.labels = copy.deepcopy(loader.labels)
        self.categorizations = copy.deepcopy(loader.classes)

        # Hyperparameter
        self.label = "subject"  # chose gender,subject or actvity
        if self.label == "gender":
            self.n_timesteps = 300
        elif self.label == "subject":
            self.n_timesteps = 600
        else:
            self.label = "activity"
            self.n_timesteps = 300
        self.label_categories = self.categorizations[self.label]
        # # Training (Hyperparameters)
        self.shuffle = True
        self.batch_size = 20
        self.epochs = 600
        # END Hyperparameter

        del loader


##############################################################################################
# Neural Network
# API
##############################################################################################


class DeepLearning(CNN2D.NNDefinition):
    def __init__(self, hyperparameter):
        super().__init__(hyperparameter)
        self.parameter: HyperParameters = hyperparameter

    def process_ydata(self, y, idx_train, idx_test):
        le = preprocessing.OneHotEncoder(categories="auto")
        y = le.fit_transform(y.reshape(-1, 1))
        y_train = y[idx_train]
        y_test = y[idx_test]
        return y_train, y_test

    def process_recurrent_xdata(self, loader, df_list, label_df,idx_train, idx_test):
        ## https://www.kaggle.com/tigurius/recuplots-and-cnns-for-time-series-classification/notebook
        ## and
        ## https://www.kaggle.com/jdarcy/introducing-ssa-for-time-series-decomposition
        def check_pics(self, transform='recurrence'):
            ### Check Pics
            if transform == 'recurrence':
                ts_to_picture = ts_to_pic
            else:
                ts_to_picture = ts_to_pic_fft
            acts = np.unique(label_df["activity"])
            fig, axs = plt.subplots(2, 3, figsize=(15, 14))
            for ax, act in zip(axs.flatten(), acts):
                i = np.where(label_df["activity"] == act)[0][0]
                ax.imshow(ts_to_picture(df_list[i])[:, :, 0])
                ax.set_title(act)
            # %%
            act = "wlk"
            subs = range(1, 7)
            # cols = ["attitude.roll","attitude.pitch","attitude.yaw","gravity.x","gravity.y","gravity.z","rotationRate.x","rotationRate.y","rotationRate.z","userAcceleration.x", "userAcceleration.y", "userAcceleration.z"]
            fig, axs = plt.subplots(2, 3, figsize=(15, 14))
            for ax, sub in zip(axs.flatten(), subs):
                i = np.where((label_df["activity"] == act) & (label_df["subject"] == sub))[0][0]
                ax.imshow(ts_to_picture(df_list[i])[:, :, 0])
                ax.set_title(sub)
            # %%
            act = "wlk"
            subs = range(1, 3)
            # cols = ["attitude.roll","attitude.pitch","attitude.yaw","gravity.x","gravity.y","gravity.z","rotationRate.x","rotationRate.y","rotationRate.z","userAcceleration.x", "userAcceleration.y", "userAcceleration.z"]
            fig, axs = plt.subplots(2, 3, figsize=(18, 14))
            for i in range(2):
                for j in range(3):
                    idx = np.where(
                        (label_df["activity"] == act) & (label_df["subject"] == subs[i])
                    )[0][j]
                    axs[i, j].imshow(ts_to_picture(df_list[idx])[:, :, 0])
                    axs[i, j].set_title(subs[i])

        def recurrence_plot(s, eps=None, steps=None):
            if eps == None:
                eps = 0.1
            if steps == None:
                steps = 10
            d = skm.pairwise.pairwise_distances(s)
            d = np.floor(d / eps)
            d[d > steps] = steps
            # Z = squareform(d)
            return d

        ## one channel/input per column
        def ts_to_pic(df):
            ft_channels = []
            for c in loader.input_signal_types:
                pic = recurrence_plot(df[c].values.reshape(-1, 1), steps=100)
                pic = np.resize(pic, pic_dims)  ## to be replaced
                ##pic = cv3.resize(img, dsize=pic_dims, interpolation=cv3.INTER_CUBIC)
                ft_channels.append(pic)
            pics = np.dstack(ft_channels)
            return pics

        #### per STFT
        ## Alternative: Use STFT

        def ts_to_pic_fft(df):
            ft_channels = []
            for c in loader.input_signal_types:
                pic = np.abs(librosa.stft(df[c].values, n_fft=60))
                pic = np.resize(pic, pic_dims)  ## to be replaced
                ##pic = cv3.resize(img, dsize=pic_dims, interpolation=cv3.INTER_CUBIC)
                ft_channels.append(pic)
            pics = np.dstack(ft_channels)
            return pics


        ## fit scaler on all values
        scaler = preprocessing.StandardScaler()
        all_values = pd.concat(df_list)
        scaler.fit(all_values[loader.input_signal_types].values)
        ### Transformation: TS -> Pic
        #### per recurrence plot
        ## from
        ## https://www.kaggle.com/tigurius/recuplots-and-cnns-for-time-series-classification#
        # modified from https://stackoverflow.com/questions/33650371/recurrence-plot-in-python
        pic_dims = (64, 64)
        check_pics('recurrence')
        plt.show()
        check_pics('fft')
        plt.show()
        n_channels = len(loader.input_signal_types)

        n_samples = len(df_list)
        X = np.zeros((n_samples, pic_dims[0], pic_dims[1], n_channels))
        for i, df in enumerate(df_list):
            if i / 10 == i // 10:
                print(f"{i} von {len(df_list)}")
            pic = ts_to_pic(df)
            #pic = ts_to_pic_fft(df)
            X[i, :, :, :] = pic

        # X = X.reshape(X.shape[0], 1, 32,32)
        # convert to float32 and normalize to [0,1]
        X = X.astype("float32")
        X /= np.amax(X)
        X_train = X[idx_train]
        X_test = X[idx_test]

        return X_train, X_test

    def process_stft_xdata(self, loader, df_list, idx_train, idx_test):
        def fft_ts(df):
            ft_channels = []
            for c in loader.input_signal_types:
                fft = np.abs(librosa.stft(df[c].values, n_fft=n_fft)).T
                ft_channels.append(fft)
            fft = np.dstack(ft_channels)
            return fft

        n_input = len(loader.input_signal_types)
        n_timesteps = self.parameter.n_timesteps
        n_samples = len(df_list)

        ## fit scaler on all values
        scaler = preprocessing.StandardScaler()
        all_values = pd.concat(df_list)
        scaler.fit(all_values[loader.input_signal_types].values)

        ### transform dataset of n_timestep with n_inputs (2D)
        ### into short fourier transformed dataset of stft-timestep fft_T, n_freq and n_inputs(3D)
        n_fft = 30  ## half minute
        ## hop length is n_fft/4
        fft_T = 1 + n_timesteps // (n_fft // 4)  ## time step of resulting stft series
        n_freq = 1 + n_fft // 2

        ## check shape
        assert fft_ts(df_list[0]).shape == (
            fft_T,
            n_freq,
            n_input,
        ), "ERROR: Not identical shape"

        X = np.zeros((n_samples, fft_T, n_freq, n_input))
        for i, df in enumerate(df_list):
            if i / 10 == i // 10:
                print(f"{i} von {len(df_list)}")
            fft = fft_ts(df)
            X[i, :, :, :] = fft
        ## fit scaler on all values
        scaler = preprocessing.StandardScaler()
        all_values = X.reshape(-1, n_input)
        scaler.fit(all_values)
        for i in range(n_samples):
            for j in range(fft_T):
                X[i, j, :, :] = scaler.transform(X[i, j, :, :])
        X_train = X[idx_train]
        X_test = X[idx_test]

        return X_train, X_test

    @timing
    def load_data(
        self
    ) -> Tuple[
        Tuple[Union[DataFrame, Series, ndarray, csr_matrix], ...],
        Tuple[Union[DataFrame, Series, ndarray, csr_matrix], ...],
        Tuple[Union[DataFrame, Series, ndarray, csr_matrix], ...],
    ]:
        # Import data
        loader = Loader()
        df_list, label_df, idx_train, idx_test = loader.motion_dataframes(
            self.parameter.label, self.parameter.n_timesteps
        )
        y_train, y_valid = self.process_ydata(
            label_df[self.parameter.label].values, idx_train, idx_test
        )
        #x_train, x_valid = self.process_stft_xdata(loader, df_list, idx_train, idx_test)
        x_train, x_valid = self.process_recurrent_xdata(loader, df_list,label_df, idx_train, idx_test)

        train_data = x_train, y_train
        valid_data = x_valid, y_valid
        test_data = pd.DataFrame(), pd.DataFrame()
        self.train_data = train_data
        self.test_data = test_data
        self.validation_data = valid_data
        return train_data, valid_data, test_data

    def setup_and_train_network(self):
        # Data
        # # Loading
        train_data, valid_data, test_data = self.load_data()

        # Model
        # # Definition
        model = self.define_model(train_data[0].shape, train_data[1].shape)

        # # Training
        model, final_metrics, label_vectors, training_history = self.train_network(
            epochs=self.parameter.epochs
        )

        return model, final_metrics, label_vectors, training_history


if __name__ == "__main__":
    # Information printout
    from tensorflow.python.client import device_lib

    logger.debug("ENVIRONMENT")
    for device in device_lib.list_local_devices():
        logger.debug(
            f"{device.name}:{device.device_type} with memory {device.memory_limit}"
        )
        logger.debug(f" {device.physical_device_desc}")

    # Define Hyperparameters
    hyperparameters = HyperParameters(model_name=model_name, loglevel=logging.DEBUG)
    # Setup Model
    neural_network = DeepLearning(hyperparameters)
    # Train model
    neural_network.setup_and_train_network()
    # Do more stuff
    # neural_network.train_network(100)
