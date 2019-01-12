import time
import os
import re


def get_time_str():
    return time.strftime("%d.%m. %H:%M:%S", time.gmtime())


def check_path(directory):
    """ Makes sure that the directory exist. """
    # Create directories
    path_trace = directory.split(os.sep)
    for i in len(path_trace):
        partial_dir = os.path.join(*(path_trace[:i+1]))
        if not os.path.exists(partial_dir):
            print("Creating directory: " + partial_dir)
            os.makedirs(partial_dir)


def dict_to_string(dictionary):
    """
    Converts a dictionary to a string.
    :param dictionary: A python dictionary
    :return: Concatenated string of all key-value pairs in the dictionary.
    """
    string = str(dictionary)
    rgx = re.compile("[':{} ]")
    string = rgx.sub("", string)

    return string


def params_to_string(params):
    """
    Converts a hyper parameter setting to a string.
    :param params: A tensorflow HParams object
    :return: Concatenated string of all key-value pairs in the parameter setting.
    """
    string = dict_to_string(params.values())
    return string


def copy_dict_to_hparams(dictionary):
    hparams = tf.contrib.training.HParams()
    for p in dictionary.keys():
        hparams.add_hparam(p, dictionary[p])
    return hparams
