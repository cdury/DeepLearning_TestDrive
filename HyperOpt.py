#!/usr/bin/python3
# Paper - https://conference.scipy.org/proceedings/scipy2013/pdfs/bergstra_hyperopt.pdf
#         https://iopscience.iop.org/article/10.1088/1749-4699/8/1/014008/pdf

# API   - Needs to import:
#           Object hyperparameters with method:
#               loss_acc, add_dict = self.run_and_get_error()
#         Members get filled (if they don't exists, they will be created):
#               self.parent_name :   filled with content of <optimization_name>
#               self.tid         :   filled with <trial id>
#               self.tb_suffix   :   filled <self.parent_name>_tid_<self.tid>
#                                       for use with tensorboard
#               self.(keys of <space>): filled with realized values of <space[key]>
from TF_Categorical import hyperparameters

import os
from hyperopt import hp
from hyperopt import fmin, tpe
from hyperopt import Trials
from hyperopt import STATUS_OK, STATUS_FAIL
from functools import partial
import numpy as np
import signal
from typing import Dict, Any
import sys
import pickle

# Even with hyperoptimazation, there are still some parameters
optimization_name = "UHR_LSTM_only"
HYPER_PATH = os.path.join("hyperopt")
HYPER_FILE = optimization_name + ".hyp"
loadFromPickle = False  # True: continue optimization / False: new optimization
num_trials = 11  # Number of trial-runs

# The key in the space must match a variable name in HyperParameters
# (has to be populated with domain knowledge)
init_space = {}
# {"random_seed": hp.choice("random_seed", (0, 1, 3, 7, 11, 33, 42, 110))}
build_space = {}
# {"size_of_hidden_layer": hp.uniform("size_of_hidden_layer", -2, 2)}
data_space = {}
learning_space = {
    "learning_rate": hp.qloguniform("learning_rate", np.log(0.0001), np.log(1), 0.0001)
}

space = {**init_space, **build_space, **data_space, **learning_space}


trials = Trials()


def objective(args: Dict[str, Any]) -> Dict[str, Any]:
    """Objective function to be minimized per hyperopt

    :param args: A realization of the parameter space "space"
    :return:     Results with this particular sets of parameter
    """
    # Amend hyperparameters
    # Optimazation Name
    hyperparameters.parent_name = optimization_name
    # Trial ID
    hyperparameters.tid = trials.tids[-1]
    hyperparameters.tb_suffix = (
        f"{hyperparameters.parent_name}_tid_{hyperparameters.tid:04}"
    )
    # Search Space
    for key, value in args.items():
        if int(value) == value:
            # convert int-numbers to int-type
            value = int(value)
        setattr(hyperparameters, key, value)

    # run with this particular hyperparameter realization
    try:
        loss_acc, additional_dict = hyperparameters.run_and_get_error()
        train_loss, train_accuracy, test_loss, test_accuracy = loss_acc
        # show brief result
        print("For", args)
        print(f"Loss    (Train/Test): {train_loss}/{test_loss}")
        print(f"Accuracy(Train/Test): {train_accuracy}/{test_accuracy}")
        print("")

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
    pickle.dump(trials, open(os.path.join(HYPER_PATH, HYPER_FILE), "wb"))


def summarize_trials() -> None:
    """ Prints a summary of the done trials

    :return:
    """
    print()
    print()
    print(
        "Trials is:", np.sort(np.array([x for x in trials.losses() if x is not None]))
    )
    for trial_dict in sorted(
        trials.trials,
        key=lambda entry: entry["result"]["true_accuracy"]
        if entry["result"]["status"] == STATUS_OK
        else 0.0,
    ):
        # for key, value in trial_dict.items():
        #    # print(key,value)
        #    pass
        result = trial_dict["result"]
        misc = trial_dict["misc"]
        if result["status"] == STATUS_OK:
            print(
                f"tid {misc['tid']}: {misc['vals']} "
                f"=>\n"
                f"        Acc {result['true_accuracy']} ({result['accuracy']})"
                f" Loss ({result['true_loss']}/{result['loss']})"
            )

        else:
            print(f"tid {misc['tid']}: {misc['vals']} => NaN")
        # saved_nn = trials.trial_attachments(trial_dict)["saved"]


def main() -> None:
    """ Conduct the hyperoptimization

    :return:
    """
    global trials
    try:
        if loadFromPickle:
            # continue optimazation with already run trials
            trials = pickle.load(open(os.path.join(HYPER_PATH, HYPER_FILE), "rb"))
        else:
            print("Starting new trials file")
    except Exception as e:
        print("Starting new trials file", e)

    best = None
    i = None
    for i in range(num_trials):
        # run optimazation once
        # ToDo: check if this ist true,and if, if it yields the same result as do the optimazation at once
        best = fmin(
            objective,
            space=space,
            algo=tpe.suggest,  # for tpe also is possible algo=partial(tpe.suggest, n_startup_jobs=1)
            max_evals=(i + 1),
            trials=trials,
        )
        # Save after each optimazation step
        save_trials()

    # end for

    # optimazation finished
    summarize_trials()
    print(f"{i} Trials => Best result was: {best}")


def signal_handler(signal, frame) -> None:
    """

    :param signal: unknown
    :param frame:  unknown
    :return:
    """
    # in case of premature interrruption
    print("Hyperoptimazation interrupted")
    # show and save state
    summarize_trials()
    save_trials()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    print(f"Start Hyperoptimazation: {optimization_name}")
    main()
