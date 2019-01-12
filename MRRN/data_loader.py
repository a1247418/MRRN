import os
import random

import numpy as np
import tensorflow as tf


class BasicGenerator:
    """
    Generator that returns one entry at a time, given a dataset and a subset of indices to consider.
    """
    def __init__(self, data_dict, indices):
        self.data = data_dict
        self.indices = indices

    def gen(self):
        n_samples = len(self.indices)
        idx = 0
        while True:
            to_yield = {k:v[self.indices[idx]] for k,v in self.data.items()}
            idx = (idx+1) % n_samples

            yield to_yield


def _load_data_from_npy(fname):
    """ Load a data set from a .npy file."""
    data_in = np.load(fname)[()]
    data = {}
    meta = {}

    for key, value in data_in.items():
        if key is not "treatment_types":
            data[key] = np.float32(value)

    to_int = ['t']
    for key in to_int:
        if key in data.keys:
            data[key] = data[key].astype('int')

    meta["treatment_types"] = np.int32(data_in["treatment_types"])
    meta['n_treatments'] = int(max(data['t'][:, 0])) + 1  # Counting control as well
    meta['dim'] = data['x'].shape[1]
    meta['n_samples'] = data['x'].shape[0]
    meta['n_pcf_samples'] = np.shape(data['s_pcf'])[2]

    return data, meta


def save_data_as_npy(data, meta, fname):
    """ Saves the data set as .npy file."""
    for key in data.keys():
        data[key] = data[key].squeeze()
    data["treatment_types"] = meta["treatment_types"]
    np.save(fname, data)


def _get_permutation_from_seed(length, seed):
    """
    Returns a permutation vector where the random seed must be specified.
    :param length: length of the vector
    :param seed:
    :return: permutation vector
    """
    permutation = [i for i in range(length)]
    if seed:
        random.Random(seed).shuffle(permutation)
    else:
        random.shuffle(permutation)
    return permutation


def _get_iterators_from_data(data_dict, meta, generator, batch_size=[None], splits=[1], do_shuffle=[False], splitting_seed=0):
    iterators = []

    n_samples = meta["n_samples"]

    permutation = _get_permutation_from_seed(n_samples, splitting_seed)

    start = 0
    for i, split in enumerate(splits):
        if i == len(splits) - 1:
            split_seq = permutation[start:]
        else:
            end = int(start + split * n_samples)
            split_seq = permutation[start:end]
            start = end
        if not batch_size[i]:
            batch_size[i] = len(split_seq)

        gen = generator(data_dict, split_seq).gen

        types = {}
        shapes = {}
        for k in data_dict.keys:
            ty = tf.int32 if data_dict[k].dtype == np.dtype('int') else tf.float32
            shape = list(np.shape(data_dict[k]))
            shape[0] = 1
            types[k] = ty
            shapes[k] = shape

        data_tensor = tf.data.Dataset.from_generator(gen, types, shapes)

        if do_shuffle[i]:
            data_tensor = data_tensor.apply(tf.data.experimental.shuffle_and_repeat(len(split_seq) + 1))
        else:
            data_tensor = data_tensor.repeat()

        data_tensor = data_tensor.batch(batch_size[i])
        data_tensor = data_tensor.prefetch(tf.contrib.data.AUTOTUNE)

        iterator = data_tensor.make_initializable_iterator()
        iterators.append(iterator)

    # Singular output for single input
    if len(splits) == 1:
        iterators = iterators[0]

    return iterators


def _get_data_file(experiment, file_number):
    """
    Assembles the data file path and name.
    :param experiment:
    :param file_number: 0 based index of the data file.
    :return: file path + name
    """
    data_path = experiment.config.data_dir
    dataset_name = experiment.config.dataset
    data_path = os.path.join(data_path, dataset_name)
    file_name = experiment.config.experiment_name
    file = data_path + os.sep + file_name + "_" + str(file_number)

    # Check whether exactly one matching file exists
    matching_files = os.listdir(file + "*")
    assert len(matching_files) == 1

    file_ending = "." + matching_files[0].split(".")[-1]

    return file + file_ending


def load_data(experiment, file_index, opt_params=None, batch_size=None,
              splits=None, do_shuffle=None, splitting_seed=None):
    """
    Returns iterators for the specified data file. Either opt_params or batch_size must
    be specified.
    :param experiment: the experimental setup
    :param file_index: the 0-based index of the data file
    :param opt_params: opt_params to take the batch sizes from
    :param batch_size: list of batch sizes
    :param splits: list of splits
    :param do_shuffle: list of boolean values indicating whether to shuffle the data
    :param splitting_seed: seed used for shuffling the data. default is the one specified in the experiment.
    :return: list of data iterators, dataset metadata
    """
    assert opt_params is not None or batch_size is not None

    if splitting_seed is None:
        splitting_seed = experiment.config.splitting_seed

    if splits is None:
        eval_fract = experiment.evaluation_fraction
        val_fract = experiment.validation_fraction
        train_fract = 1 - eval_fract - val_fract
        splits = [train_fract, val_fract, eval_fract]
    if batch_size is None:
        batch_size = [opt_params.batch_size]*3
    if do_shuffle is None:
        do_shuffle = [True, True, False]

    file = _get_data_file(experiment, file_index)
    file_ending = file.split(".")[-1]

    iterators = None

    if file_ending == "npy":
        data_dict, meta = _load_data_from_npy(file)
        iterators = _get_iterators_from_data(data_dict, meta, BasicGenerator, batch_size=batch_size,
                                 splits=splits, do_shuffle=do_shuffle, splitting_seed=splitting_seed)
    else:
        print("Unknown file format '{file_ending}'".format(file_ending=file_ending))

    return iterators, meta
