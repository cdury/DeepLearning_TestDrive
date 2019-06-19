import os
from functools import wraps
from time import process_time, perf_counter, strftime, gmtime

# typing imports
from typing import Tuple, List, Any, Union, Optional
from numpy import ndarray
from pandas import DataFrame, Series

# keras imports
from tensorflow.contrib.keras.api.keras.models import Model
from tensorflow.contrib.keras.api.keras.callbacks import (
    TensorBoard,
    ModelCheckpoint,
    EarlyStopping,
)


def timing(method):
    @wraps(method)
    def wrap(*args, **kw):
        wall_ts = perf_counter()
        pu_ts = process_time()
        return_value = method(*args, **kw)
        wall_te = perf_counter()
        pu_te = process_time()
        print(
            f"Method:{method.__name__}"
            f" took {wall_te-wall_ts:2.4f}sec ({pu_te-pu_ts:2.4f}sec)"
        )
        return return_value

    return wrap


class BaseParameters:
    def __init__(self, model_name, loglevel):
        # Technical & HyperOpt
        self.model_name = model_name
        self.parent_name = ""  # RunName of the optimization run when using hyperopt
        self.tid = None  # Trial ID when using hyperopt
        self.tb_suffix = ""  #
        self.timestamp = strftime("%y%m%d%H%M", gmtime())

        # Filesystem & Paths
        home_path = os.getcwd()
        base_ts_name = model_name + "_" + self.timestamp
        self.log_path = os.path.join(home_path, "log")

        # # data
        self.data_path = os.path.join(home_path, "data")

        # # model
        self.model_path = os.path.join(home_path, "models")
        self.model_dir = os.path.join(self.model_path, base_ts_name)
        if not os.path.isdir(self.model_dir):
            os.mkdir(self.model_dir)

        # # tensorboard
        self.tensorboard_path = os.path.join(home_path, "tensorboard")
        self.tensorboard_dir = os.path.join(self.tensorboard_path, base_ts_name)

        # Logging
        # # Internal Tensorflow logging
        # 0: All Msg 1: No INFO 2: No INFO & WARNING 3: No INFO, WARNING & ERROR
        if isinstance(loglevel, int):
            str_loglevel = str(loglevel)
        elif isinstance(loglevel, str):
            str_loglevel = loglevel
            loglevel = int(loglevel)
        else:
            str_loglevel = "0"
            loglevel = 0
        self.loglevel = loglevel
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = str_loglevel
        # DANGER Unknown
        # # MKL_THREADING_LAYER
        # https://software.intel.com/en-us/mkl-linux-developer-guide-dynamically-selecting-the-interface-and-threading-layer
        # os.environ["MKL_THREADING_LAYER"] = 'GNU' # "INTEL","SEQUENTIAL","GNU","PGI","TBB"

        # # Keras model
        self.show_graphs = True
        self.eval_verbosity = 0  # 0: quiet // 1: update_messagse
        self.fitting_verbosity = 2  # 0: quiet // 1: animation // 2: summary

        # Hyperparameter
        # # Data

        # # Modell (set in code)
        self.n_input = None  # Set in code
        self.n_classes = None  # Number of classification classes (set in code)

        # # Callbacks
        self.callback_verbosity = 1  # 0: quiet // 1: update_messagse
        # # # Checkpoint TensorBoard
        self.tb_update_freq = "epoch"
        self.tb_write_graph = True
        self.tb_write_images = True
        # # # Checkpoint ModelCheckpoint
        self.save_every_epochs = 1
        # # Training (Hyperparamters)
        self.monitor = "val_loss"  # "val_acc"
        self.mode = "auto"  # ""min"  # "max"
        self.patience = 5
        self.batch_size = 0
        self.epochs = 1000


class BaseNN:
    def __init__(self, hyperparameter):
        self.parameter: BaseParameters = hyperparameter
        self.test_data = None
        self.validation_data = None

    # Following methods must be implemented individually
    def load_data(
        self
    ) -> Tuple[
        Tuple[Union[DataFrame, Series, ndarray], ...],
        Tuple[Union[DataFrame, Series, ndarray], ...],
        Tuple[Union[DataFrame, Series, ndarray], ...],
    ]:
        pass

    def define_model(self, data_shape: Tuple[int, ...]) -> Model:
        # Abstract function / Interface ?
        pass

    def train_model(
        self,
        model: Model,
        train_data: Tuple[Union[DataFrame, Series, ndarray], ...],
        validation_data: Tuple[Union[DataFrame, Series, ndarray], ...],
    ) -> List[Any]:
        pass

    # Following methods can be used
    def get_callbacks(self):
        # Callbacks
        # for Tensorboard evaluation
        tensorboard = TensorBoard(
            log_dir=self.parameter.tensorboard_dir,
            update_freq=self.parameter.tb_update_freq,
            write_graph=self.parameter.tb_write_graph,
            write_images=self.parameter.tb_write_images,
            histogram_freq=0,
            write_grads=False,
        )
        # for saving network with weights
        checkpoint = ModelCheckpoint(
            filepath=os.path.join(
                self.parameter.model_dir, "weights.{epoch:03d}-{val_loss:.2f}.hdf5"
            ),
            monitor=self.parameter.monitor,
            save_best_only=True,
            mode=self.parameter.mode,
            verbose=self.parameter.callback_verbosity,
            period=self.parameter.save_every_epochs,
        )
        # for early termination
        earlyterm = EarlyStopping(
            monitor=self.parameter.monitor,
            mode=self.parameter.mode,
            patience=self.parameter.patience,
            restore_best_weights=True,
            verbose=self.parameter.callback_verbosity,
            min_delta=0.001,
        )
        return tensorboard, checkpoint, earlyterm

    @timing
    def calc_categorical_accuracy(self, model, test_data):
        final_metrics = {}
        x_test, y_test = test_data
        score = model.evaluate(x_test, y_test, verbose=self.parameter.eval_verbosity)
        for name, value in zip(model.metrics_names, score):
            print(f"{name}: {value}")
            final_metrics[name] = value
        return final_metrics

    @timing
    def is_vs_should_categorical(self, model, test_data):
        x_test, y_test = test_data
        predictions = model.predict(x_test)
        given = y_test.argmax(1)
        return predictions.argmax(1), given
