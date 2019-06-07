import os
from typing import Tuple, List, Any, Union, Optional
from numpy import ndarray
from pandas import DataFrame, Series
from functools import wraps
from time import process_time, perf_counter, strftime, gmtime

from tensorflow.contrib.keras.api.keras.models import Model


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
        self.timestamp = strftime("%y%m%d%H%M", gmtime())
        self.base_parameter = "base"
        self.parent_name = ""  # RunName of the optimization run when using hyperopt
        self.tid = None  # Trial ID when using hyperopt
        self.tb_suffix = ""  #

        # Filesystem & Paths
        home_path = os.getcwd()
        base_ts_name = model_name + "_" + self.timestamp

        # # data
        data_path = os.path.join(home_path, "data")

        # # model
        model_path = os.path.join(home_path, "models")
        self.model_dir = os.path.join(model_path, base_ts_name)
        if not os.path.isdir(self.model_dir):
            os.mkdir(self.model_dir)

        # # tensorboard
        tensorboard_path = os.path.join(home_path, "tensorboard")
        self.tensorboard_dir = os.path.join(tensorboard_path, base_ts_name)

        # Logging
        self.callback_verbosity = 0  # 0: quiet // 1: update_messagse
        self.eval_verbosity = 0  # 0: quiet // 1: update_messagse
        self.fitting_verbosity = 0  # 0: quiet // 1: animation // 2: summary
        self.log_dir = os.path.join(home_path, "log")
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
