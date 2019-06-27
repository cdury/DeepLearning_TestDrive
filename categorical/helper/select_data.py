import numpy as np
import pandas as pd
from typing import Union

#### Batch extractor
# We perform batchwise training, therefore we need a function that supplies the batches to the training algorithem:
def extract_batch_size(
    _train: Union[pd.DataFrame, np.array], step: int, batch_size: int
) -> np.array:
    """ Function to fetch a "batch_size" amount of data from "(X|y)_train" data.

    :param _train:      data with method "shape". Axis 0 denotes the individual training sets
    :param step:        number of which batch has to be extracted (i.e. 1; first batch ,....)
    :param batch_size:  number of training sets per batch
    :return:            extracted batch as numpy-Array
    """

    shape = list(_train.shape)
    shape[0] = batch_size
    batch_s = np.empty(shape)

    for i in range(batch_size):
        # Loop index
        index = ((step - 1) * batch_size + i) % len(_train)
        batch_s[i] = _train[index]

    return batch_s
