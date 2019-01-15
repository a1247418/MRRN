import sys
import utils

import tensorflow as tf

import config_loader
import data_loader
import preprocess
import learning_routines
import postprocess


def run(experiment):
    #  Check whether all required paths exist
    for key in experiment.config.keys():
        if key.endswith("_dir"):
            utils.check_path(experiment.config[key])

    n_data_files = experiment.n_evaluations

    for i in range(n_data_files):
        tf.reset_default_graph()

        iterators, meta = preprocess.preprocess(experiment, i)

        tr_iter, val_iter, eval_iter = iterators

        # TODO: structure hierarchy of file/model/preproc&core
        #learning_routines.train_and_evaluate(experiment, experiment.mod, meta, run_id, tr_iter, val_iter, eval_iter=None, save=False, propensity_params=None)

        # TODO: postprocess


if __name__ == "__main__":
    abort = False
    print('Arguments: experiment_name option_to_overwrite:value')

    if len(sys.argv) <= 1:
        print("No experiment specified.")
        abort = True
    else:
        experiment_name = sys.argv[1]
        options_to_overwrite = []
        for i in range(2, len(sys.argv)):
            argument = str(sys.argv[i])

            if not ":" in argument:
                print("Options to overwrite must have formant [option:value]")
                abort = True
                break
            else:
                argument_parts = argument.split(":")
                options_to_overwrite.append(argument_parts)

        if not abort:
            experiment = config_loader.load_experiment(experiment_name, options_to_overwrite)
            run(experiment)

    if abort:
        print("Aborting")
    else:
        print("Done")
