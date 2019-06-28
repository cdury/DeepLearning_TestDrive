#!/usr/bin/python3
# Paper - https://conference.scipy.org/proceedings/scipy2013/pdfs/bergstra_hyperopt.pdf
#         https://iopscience.iop.org/article/10.1088/1749-4699/8/1/014008/pdf

# API   - Needs to import object "hyperparameters":
#         Members get filled (if they don't exists, they will be created):
#               self.parent_name :   filled with content of <optimization_name>
#               self.tid         :   filled with <trial id>
#               self.tb_suffix   :   filled <self.parent_name>_tid_<self.tid>
#                                       for use with tensorboard
#               self.(keys of <space>): filled with realized values of <space[key]>
#
#       - Needs to import function "run_and_get_error":
#               loss_acc, add_dict = run_and_get_error(hyperparameters)
__version__ = "1.0"
# general imports
import os
import sys
import logging
import time
import signal
import pickle
from logging.config import dictConfig

# typing imports
from typing import Dict, Any

# hyperopt imports
from hyperopt import fmin, tpe
from hyperopt import Trials
from hyperopt import STATUS_OK, STATUS_FAIL

# parameter import
from categorical.Motionsense.CNN1d.hyperParams import (
    hyper_opt_name,
    model_name,
    num_trials,
    Network,
    hyper_param_space,
    run_and_get_error,
    load_hyper_params_from_pickle,
)

# technicalities
HYPER_PATH = os.path.join(os.getcwd(),"categorical",model_name)
HYPER_BASE = (
    hyper_opt_name + "_" + time.strftime("%y%m%d%H%M", time.gmtime())
)  # Unique name of a hyperoptimization run
HYPER_FILE = os.path.join(
    HYPER_PATH, HYPER_BASE + ".pkl"
)  # Pickle-Filename of the used hyperparameters

# Global object
trials = Trials()
loglevel = logging.INFO
dictConfig(
    dict(
        version=1,
        formatters={
            "f": {"format": "%(asctime)s %(name)-8s %(levelname)-6s %(message)s"}
        },
        handlers={
            "h": {"class": "logging.StreamHandler", "formatter": "f", "level": loglevel}
        },
        root={"handlers": ["h"], "level": loglevel},
    )
)
logger = logging.getLogger("HYPER")


def objective(args: Dict[str, Any]) -> Dict[str, Any]:
    """Objective function to be minimized per hyperopt

    :param args: A realization of the parameter space "space"
    :return:     Results with this particular sets of parameter
    """
    logger.info(f"Trial {trials.tids[-1]} : {args}")
    # Instantiate hyperparameters
    hyperparameters = Network.HyperParameters(
        model_name=f"{hyper_opt_name}_{trials.tids[-1]:04}",
        loglevel=logging.ERROR,
        parent_name=HYPER_BASE,
    )
    # Amend hyperparameters
    # Trial ID
    hyperparameters.tid = trials.tids[-1]
    hyperparameters.tb_suffix = (
        f"{hyperparameters.parent_name}_tid_{hyperparameters.tid:04}"
    )
    # Keras(TF) - Logging / Verbosity
    if loglevel <= logging.DEBUG:
        hyperparameters.fitting_verbosity = 2
    else:
        hyperparameters.fitting_verbosity = 0
    # Search Space
    for key, value in args.items():
        if int(value) == value:
            # convert int-numbers to int-type
            value = int(value)
        setattr(hyperparameters, key, value)

    # run with this particular hyperparameter realization
    try:
        loss_acc, additional_dict = run_and_get_error(
            hyperparameters, return_model=False
        )
        train_loss, train_accuracy, test_loss, test_accuracy = loss_acc
        # show brief result
        logger.info(
            f"Trial {trials.tids[-1]} : Loss    (Train/Test): {train_loss}/{test_loss}"
        )
        logger.info(
            f"Trial {trials.tids[-1]} : Accuracy(Train/Test): {train_accuracy}/{test_accuracy}"
        )

        # return to optimazation loop
        return {
            # built in keys
            "loss": train_loss,
            "true_loss": test_loss,
            "status": STATUS_OK,
            # custom keys
            "accuracy": train_accuracy,
            "true_accuracy": test_accuracy,
            # special key
            "attachment": additional_dict,
        }
    except Exception as e:
        return {
            # built in keys
            "status": STATUS_FAIL,
            # special key
            "attachment": {"exception": str(e)},
        }


def save_trials() -> None:
    """ Save the done trials for continuation of the optimization or future analysy

    :return:
    """
    pickle.dump(trials, open(HYPER_FILE, "wb"))


def summarize_trials() -> None:
    """ Prints a summary of the done trials

    :return:
    """
    for trial_dict in sorted(
        trials.trials,
        key=lambda entry: entry["result"]["true_accuracy"]
        if entry["result"]["status"] == STATUS_OK
        else 0.0,
    ):
        for key, value in trial_dict.items():
            logger.debug(f"trial_dict {key}:{value}")
            pass
        result = trial_dict["result"]
        misc = trial_dict["misc"]
        if result["status"] == STATUS_OK:
            logger.info(f"tid {misc['tid']}: {misc['vals']} ")
            logger.info(f"tid {misc['tid']}:  => Acc {result['true_accuracy']} ({result['accuracy']})")
            logger.info(f"tid {misc['tid']}:  => Loss ({result['true_loss']}/{result['loss']})")

        else:
            logger.info(f"tid {misc['tid']}: {misc['vals']}")
            logger.info(f"tid {misc['tid']}:  => NaN")
        # saved_nn = trials.trial_attachments(trial_dict)["saved"]


def main() -> None:
    """ Conduct the hyperoptimization

    :return:
    """
    global trials
    try:
        if load_hyper_params_from_pickle:
            logger.info(f"Continuing trials file {HYPER_FILE}")
            # continue optimazation with already run trials
            trials = pickle.load(open(HYPER_FILE, "rb"))
        else:
            logger.info(f"Starting new trials file {HYPER_FILE}")
    except Exception as e:
        logger.error("Starting new trials file", exc_info=True)

    best = None
    i = None
    for i in range(num_trials):
        # run optimazation once
        best = fmin(
            objective,
            space=hyper_param_space,
            algo=tpe.suggest,  # for tpe also is possible algo=partial(tpe.suggest, n_startup_jobs=1)
            max_evals=(i + 1),
            trials=trials,
            verbose=0,
            show_progressbar=False,
        )
        # Save after each optimazation step
        save_trials()

    # end for

    # optimazation finished
    summarize_trials()
    logger.info(f"{i+1} Trials => Best result was: {best}")


def signal_handler(signal, frame) -> None:
    """

    :param signal: unknown
    :param frame:  unknown
    :return:
    """
    # in case of premature interrruption
    logger.warning("Hyperoptimazation interrupted")
    # show and save state
    summarize_trials()
    save_trials()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    logger.info(f"Start Hyperoptimazation: {hyper_opt_name}")
    main()
