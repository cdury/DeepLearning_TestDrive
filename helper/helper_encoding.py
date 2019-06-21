import numpy as np
import pandas as pd
from typing import Union

from tensorflow.python.keras.utils import to_categorical

#### One hot encoding
# To train with the classification we represent the labe as on hot encoded vector:
def one_hot(y_: Union[pd.DataFrame, np.array], n_classes: int = None) -> np.array:
    """ Function to encode neural one-hot output labels from number indexes
    # e.g.:
    # one_hot(y_=[[5], [0], [3]], n_classes=6):
    #     return [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]

    :param y_: Labels vector
    :param n_classes: Number of classes
    :return:  One hot encoding numpy array
    """

    y_ = y_.reshape(len(y_))
    one_hot_encoded =  to_categorical(y_, n_classes, dtype=np.int32)
    return one_hot_encoded
    # return np.eye(n_classes)[np.array(y_, dtype=np.int32)]  # Returns FLOATS
