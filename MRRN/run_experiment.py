import sys
import utils

import config_loader
import preprocess
import process
import postprocess


def run(experiment):
    #  Check whether all required paths exist
    for key in experiment.config.keys():
        if key.endswith("_dir"):
            utils.check_path(experiment.config[key])

    preprocess.preprocess(experiment)

    results = process.train_and_evaluate(experiment)

    # TODO postprocess


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
