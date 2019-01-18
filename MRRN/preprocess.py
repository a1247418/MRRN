import os.path

import tensorflow as tf

import utils
import matching
import matching_def
import data_loader
import learning_routines
import loss_def


def train_propensity_model(experiment, tr_iter, val_iter, meta, run_id):
    """
    Trains and saves a propensity model, if there does not already exist a saved model.
    :param experiment: experiment setup
    :param tr_iter: iterator over training data
    :param val_iter: iterator over validation data
    :param meta: meta data
    :param run_id: 0-based file index
    :return:
    """
    # this assumes an already optimized params set
    params = experiment.propensity_model[0].model_params

    # Only train a new propensity model if it does not exist already
    prop_path = utils.assemble_model_path(experiment, "prop", run_id)
    if tf.train.latest_checkpoint(prop_path):
        print("Found propensity model... skipping training.")
    else:
        print("Did not find propensity model in: %s. Training new one." % prop_path)
        learning_routines.train_and_evaluate(experiment, experiment.propensity_model[0], meta, run_id, tr_iter, val_iter,
                                         save=True, is_propensity_model=True)


def clac_matching_estimates(experiment, tr_iter, run_id):
    """
    Calculates matching estimates if they are not saved yet, and saves them.
    :param experiment: experiment setup
    :param tr_iter: iterator over training data
    :param run_id: 0-based file index
    :return:
    """
    # Get matching estimates
    losses = experiment.additional_losses_to_record.split(',')
    for model in experiment.models:
        losses += model.model_params.train_loss.split(',')
    losses = list(set(losses))

    for loss in losses:
        if loss not in [l.value for l in loss_def.RegisteredLosses]:
            matching_path = utils.assemble_matching_path(experiment, loss, run_id)
            exists_already = os.path.isfile(matching_path + "matching.npy")

            if not exists_already:
                matching_setup = matching_def.matching_from_string(loss)
                iterator_all, meta = data_loader.load_data(experiment, run_id, batch_size=[None], splits=[1],
                                                           do_shuffle=[0])
                matching.produce_matching(experiment, iterator_all, tr_iter, meta["treatment_types"], matching_setup,
                                          run_id, save=True)


def preprocess(experiment):
    n_data_files = experiment.n_evaluations

    for i in range(n_data_files):
        print(utils.get_time_str() + "Prepocessing file {idx}/{all}".format(idx=i+1, all=n_data_files))

        tf.reset_default_graph()

        iterators, meta = data_loader.load_data(experiment, i, batch_size=[None, None, None],
                                                do_shuffle=[False, False, False])

        # Train propensity model
        train_propensity_model(experiment, iterators[0], iterators[1], meta, i)

        # Calculate matching estimates
        print(utils.get_time_str() + " Matching...")
        clac_matching_estimates(experiment, iterators[0], i)
        print(utils.get_time_str() + " Done matching.")