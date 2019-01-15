import tensorflow as tf

from benchmark_models import NN, OLS, SplitRegressor
from MRRN_model import MRRN
from propensity_model import MultipleNN


def str2model(model_string):
    model_class = None
    if model_string == "NN":
        model_class = NN
    elif model_string == "OLS":
        model_class == OLS
    elif model_string == "OLS_split":
        model_class == SplitRegressor
    elif model_string == "MRRN":
        model_class == MRRN
    elif model_string == "MultipleNN":
        model_class == MultipleNN

    assert model_class is not None

    return model_class


def safe_sqrt(x):
    return tf.sqrt(tf.clip_by_value(x, 1e-10, np.inf))

