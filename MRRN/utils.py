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


def assemble_model_path(experiment, model_name, run_id):
    """
    Returns the path to the model saves.
    :param experiment: experimental setup
    :param run_id: 0-based index of the data file
    :return: model path
    """
    prop_path = experiment.config.saves_dir + model_name + "_" + experiment.config.dataset + "_" + \
        experiment.config.experiment_name + "_seed" + experiment.config.splitting_seed + "_" + run_id + os.sep

    return prop_path


def assemble_matching_path(experiment, matching, run_id):
    """
    Returns the path to the matching saves.
    :param experiment: experimental setup
    :param run_id: 0-based index of the data file
    :return: matching path
    """
    prop_path = experiment.config.saves_dir + matching + "_" + experiment.config.dataset + "_" + \
        experiment.config.experiment_name + "_seed" + experiment.config.splitting_seed + "_" + run_id + os.sep

    return prop_path


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
