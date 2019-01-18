import tensorflow as tf

import data_loader
import learning_routines


def train_and_evaluate(experiment):
    n_data_files = experiment.n_evaluations

    outputs = {}

    for file_idx in range(n_data_files):
        file_key = "file{idx}".format(idx=file_idx)
        outputs[file_key] = {}
        for model_idx in range(len(experiment.models)):
            model_key = experiment.models[model_idx].model_name
            tf.reset_default_graph()

            model_setup = experiment.models[file_idx]
            if "propensity_model" in experiment.keys():
                propensity_params = experiment.propensity_model[0].model_params
            else:
                propensity_params = None

            iterators, meta = data_loader.load_data_and_matchings(experiment, file_idx, opt_params=model_setup.opt_params)
            tr_iter, val_iter, eval_iter = iterators

            outputs = learning_routines.train_and_evaluate(experiment, model_setup, meta, file_idx, tr_iter, val_iter, eval_iter,
                                                 propensity_params)
            outputs[file_key][model_key] = outputs

    return outputs


def hparam_search(experiment):
    # TODO
    yield