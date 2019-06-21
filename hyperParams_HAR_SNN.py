# general imports
import numpy as np

# hyperopt imports
from hyperopt import hp

# Even with hyperoptimization, there are still some parameters
from Categorical import run_and_get_error, HyperParameters  # Model to be optimized
hyper_opt_name = "HAR_SNN"  # Name of optimization run <NAME_OF_DATASET>_<NAME_OF_NETWORK>
load_hyper_params_from_pickle = False  # True: continue optimization / False: new optimization
num_trials = 3  # Number of trial-runs

# The key in the space must match a variable name in HyperParameters
# (has to be populated with domain knowledge)
init_space = {
}
# {
#  "random_seed": hp.choice("random_seed", (0, 1, 3, 7, 11, 33, 42, 110)),
#  }
build_space = {
    "n_hidden_1": hp.qloguniform("n_hidden_1", np.log(100), np.log(10000), 1),
    "n_hidden_2": hp.qloguniform("n_hidden_2", np.log(50), np.log(5000), 1),
    "n_hidden_3": hp.qloguniform("n_hidden_3", np.log(25), np.log(2500), 1),
}
# {"size_of_hidden_layer": hp.uniform("size_of_hidden_layer", -2, 2)}
data_space = {}
learning_space = {
    "learning_rate": hp.qloguniform("learning_rate", np.log(0.0001), np.log(1), 0.0001),
    "lambda_loss_amount": hp.qloguniform(
        "lambda_loss_amount", np.log(0.0000001), np.log(0.1), 0.0000001
    ),
    "batch_size": hp.quniform("batch_size", 100, 6000, 1),
}

hyper_param_space = {**init_space, **build_space, **data_space, **learning_space}
