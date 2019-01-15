import tensorflow as tf

import utils
import matching
import matching_def
import data_loader
import learning_routines
import loss_def


def train_propensity_model(experiment, tr_iter, val_iter, meta, run_id):
    # this assumes an already optimized params set
    params = experiment.propensity_model[0].model_params

    # Only train a new propensity model if it does not exist already
    prop_path = utils.asseble_prop_path(experiment, run_id)
    if tf.train.latest_checkpoint(prop_path):
        print("Found propensity model... skipping training.")
    else:
        print("Did not find propensity model in: %s. Training new one." % prop_path)
        learning_routines.train_and_eval(experiment, experiment.propensity_model[0], meta, run_id, tr_iter, val_iter,
                                         save=True)

    return params


def preprocess(experiment, run_id):
    iterators, meta = data_loader.load_data(experiment, run_id, batch_size=[None,None,None],
                                         do_shuffle=[False, False, False])

    # Train propensity model
    train_propensity_model(experiment, iterators[0], iterators[1], meta, i)

    # Get matching estimates
    matching_results = {}
    group_defs_all = {}
    losses = experiment.additional_losses_to_record.split(',') + experiment.opt_params.train_loss.split(',')
    for loss in losses:
        if loss not in [l.value for l in loss_def.RegisteredLosses]:
            matching_setup = matching_def.matching_from_string(loss, experiment)
            iterator_all, meta = data_loader.load_data(experiment, run_id, batch_size=[None], splits=[1],
                                                       do_shuffle=[0])
            iterator_all = iterator_all[0]
            y_hat, group_defs = matching.produce_matching(experiment, iterator_all, iterators[0],
                                                              matching_setup, meta["treatment_types"], i,
                                                              save=False)

            group_defs_all["group_"+loss] = group_defs
            matching_results[loss] = y_hat

    iterators, meta = data_loader.load_data(experiment, i, batch_size=[None,None,None],
                                         do_shuffle=None, repeat=[True, True, False], additional_columns=matching_results)

    meta = meta.update(group_defs_all)

    return iterators, meta
