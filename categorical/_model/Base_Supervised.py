import os
import logging
import numpy as np

from time import strftime, gmtime
from logging.config import dictConfig
from categorical._helper.profiling import timing

# typing imports
from typing import Tuple, List, Any, Union, Optional
from numpy import ndarray
from pandas import DataFrame, Series

# keras imports
from tensorflow.python.keras.api._v2 import keras

# # Model
Model = keras.models.Model
# # Callback
TensorBoard = keras.callbacks.TensorBoard
ModelCheckpoint = keras.callbacks.ModelCheckpoint
EarlyStopping = keras.callbacks.EarlyStopping

# Global object
loglevel_this = logging.DEBUG
dictConfig(
    dict(
        version=1,
        formatters={
            "f": {"format": "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"}
        },
        handlers={
            "h": {
                "class": "logging.StreamHandler",
                "formatter": "f",
                "level": loglevel_this,
            }
        },
        root={"handlers": ["h"], "level": loglevel_this},
    )
)
logger = logging.getLogger("Base_Supervised_Categorical")


class BaseParameters:
    def __init__(self, model_name, dir_path, loglevel, parent_name=None):
        # Technical & HyperOpt
        self.model_name = model_name
        self.parent_name = (
            parent_name
        )  # RunName of the optimization run when using hyperopt
        self.tid = None  # Trial ID when using hyperopt
        self.tb_suffix = ""  #
        self.timestamp = strftime("%y%m%d%H%M", gmtime())

        # Filesystem & Paths
        home_path = os.getcwd()
        base_ts_name = model_name + "_" + self.timestamp
        self.log_path = os.path.join(
            home_path, "log", parent_name if parent_name else model_name
        )

        # # data
        self.data_path = os.path.join(home_path, "data")

        # # _model
        self.model_path = os.path.join(
            home_path,
            "categorical",
            dir_path,
            parent_name if parent_name else "",  # model_name
        )
        self.model_dir = os.path.join(self.model_path, base_ts_name)
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)

        # # tensorboard
        self.tensorboard_path = os.path.join(
            home_path, "tensorboard", parent_name if parent_name else model_name
        )
        self.tensorboard_dir = os.path.join(self.tensorboard_path, base_ts_name)

        # Logging
        # # Internal Tensorflow logging
        # 0: All Msg 1: No INFO 2: No INFO & WARNING 3: No INFO, WARNING & ERROR
        if loglevel <= logging.DEBUG:
            str_loglevel = "0"
        elif logging.INFO <= loglevel <= logging.WARNING:
            str_loglevel = "1"
        elif loglevel == logging.ERROR:
            str_loglevel = "2"
        else:
            str_loglevel = "3"
        self.loglevel = loglevel
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = str_loglevel
        # DANGER Unknown
        # # MKL_THREADING_LAYER
        # https://software.intel.com/en-us/mkl-linux-developer-guide-dynamically-selecting-the-interface-and-threading-layer
        # os.environ["MKL_THREADING_LAYER"] = 'GNU' # "INTEL","SEQUENTIAL","GNU","PGI","TBB"

        # # Keras _model
        self.show_graphs = True
        self.eval_verbosity = 0  # 0: quiet // 1: update_messagse
        self.fitting_verbosity = 2  # 0: quiet // 1: animation // 2: summary

        # Hyperparameter
        # # Data

        # # Modell (set in code)
        self.input_shape = None  # Set in code
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
        self.patience = 10
        # # Results
        self.trained_epochs = 0
        # # Placeholder (need to be set in child object)
        self.shuffle = None
        self.batch_size = None
        self.epochs = None


class BaseNN:
    def __init__(self, hyperparameter):
        self.parameter: BaseParameters = hyperparameter
        self.model = None
        self.train_data = None
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
    def calc_categorical_accuracy(self, model, non_train_data, add_data=None):
        final_metrics = {}
        x, y = non_train_data
        score = model.evaluate(x, y, verbose=self.parameter.eval_verbosity)
        for name, value in zip(model.metrics_names, score):
            logger.info(f"{name}: {value}")
            final_metrics[name] = value

        # Accuracy calculation by hand (same result than accuracy of _model.evaluate)
        # import sklearn.metrics as skm
        # y_pred = _model.predict(x)
        # final_metrics['sklearn_acc'] = skm.accuracy_score(y.argmax(axis=1), y_pred.argmax(axis=1))

        return final_metrics

    @timing
    def is_vs_should_categorical(self, model, non_train_data):
        x, y = non_train_data
        index = len(y.shape) - 1
        predictions = model.predict(x)
        given = np.asarray(y.argmax(index)).reshape(-1)
        predictions = np.asarray(predictions.argmax(index)).reshape(-1)
        return predictions, given
