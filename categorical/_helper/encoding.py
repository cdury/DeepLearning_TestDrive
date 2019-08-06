import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

from typing import Union, Tuple, Optional

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

    if not n_classes:
        n_classes = np.max(y_) + 1
    y_ = y_.reshape(len(y_))
    one_hot_encoded = to_categorical(y_, n_classes, dtype=np.int32)
    return one_hot_encoded
    # return np.eye(n_classes)[np.array(y_, dtype=np.int32)]  # Returns FLOATS


def one_hot_sparse(values: np.array, categories="auto", textlabels=False, sparse=False) -> Tuple[csr_matrix, Optional[LabelEncoder]]:
    """

    :param values:  Labels vector
    :param categories: "auto" or list of possible labels
    :param textlabels: If labels are text or not
    :param sparse: wether returned one-hot is a np.array or a sparse matrix
    :return: One hot encoding in a sparse matrix, LabelEncoder for text labels

    :Note:
        # invert one-hot encoded
        inverted = label_encoder.inverse_transform([np.argmax(onehot_encoded[0, :])])
    """

    if textlabels:
        # integer encoding
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(values)
        # binary encode
        onehot_encoder = OneHotEncoder(categories=categories, sparse=sparse,dtype=np.int32)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        return onehot_encoded, label_encoder
    else:
        label_encoder = None
        integer_encoded = values
        # binary encode
        onehot_encoder = OneHotEncoder(categories=categories, sparse=sparse,dtype=np.int32)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        return onehot_encoded





