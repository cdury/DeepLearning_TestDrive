# general imports
import numpy as np

# hyperopt imports
from hyperopt import hp

# Even with hyperoptimization, there are still some parameters
from Categorical import Network  # Model to be optimized
from Categorical import run_and_get_error  # Model to be optimized

hyper_opt_name = (
    "SIMPLENN_MLP"
)  # Name of optimization run <NAME_OF_DATASET>_<NAME_OF_NETWORK>
model_name = Network.model_name
load_hyper_params_from_pickle = (
    False
)  # True: continue optimization / False: new optimization
num_trials = 100  # Number of trial-runs

# The key in the space must match a variable name in HyperParameters
# (has to be populated with domain knowledge)
init_space = {
  "random_seed": hp.choice("random_seed", (0, 1, 3, 7, 11, 33, 42, 110)),
  }
build_space = {
    "number_of_layers": hp.quniform("number_of_layers", 2, 10,1),
    # "dropout" : hp.quniform("dropout", 0, 1, 0.01),
    "layer_size_start": hp.quniform("layer_size_start", 1, 2,1),
    "layer_size_decrease" :  hp.uniform("layer_size_decrease", 1, 5)
}
data_space = {}
learning_space = {
    "learning_rate": hp.qloguniform("learning_rate", np.log(0.0001), np.log(1), 0.0001),
    "lambda_loss_amount":hp.qloguniform("lambda_loss_amount", np.log(0.0001), np.log(1), 0.0001)
}

hyper_param_space = {**init_space, **build_space, **data_space, **learning_space}