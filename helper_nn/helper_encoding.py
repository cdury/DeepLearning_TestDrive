import numpy as np
import pandas as pd
from typing import Union

#### One hot encoding
# To train with the classification we represent the labe as on hot encoded vector:
def one_hot(y_: Union[pd.DataFrame, np.array], n_classes: int) -> np.array:
    """ Function to encode neural one-hot output labels from number indexes
    # e.g.:
    # one_hot(y_=[[5], [0], [3]], n_classes=6):
    #     return [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]

    :param y_: Labels vector
    :param n_classes: Number of classes
    :return:  One hot encoding numpy array
    """


    y_ = y_.reshape(len(y_))
    return np.eye(n_classes)[np.array(y_, dtype=np.int32)]  # Returns FLOATS
