import os

from dotmap import DotMap


def _file2dict(file_path):
    dictionary = DotMap()
    sub_dict = None
    write_to_subdict = False

    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()

            if line == "":
                continue

            # Start list of dictionaries
            if line.endswith(":{"):
                key = line[:-2]
                write_to_subdict = True
                if key not in dictionary:
                    dictionary[key] = []
                dictionary[key].append(DotMap())
                sub_dict = dictionary[key][-1]
                continue
            # End list of dictionaries
            elif line == "}":
                write_to_subdict = False
                continue
            # Add field to dictionary
            else:
                key, value = line.split(":")
                key = key.strip()
                value = value.strip()
                value_type = key[0]
                if value_type == 'i':
                    value_type = int
                elif value_type == 'f':
                    value_type = float
                elif value_type == 'b':
                    value_type = lambda a: a == "1" or a == "True"
                elif value_type == 's':
                    value_type = str
                else:
                    print("Unknown value type: {type} in {file}".format(type=value, file=file_path))

                key = key[2:]

                # Detect lists
                if "[" in value:
                    value = value.replace('[', '').replace(']', '')
                    values = value.split(',')
                    value = [value_type(val.strip()) for val in values]
                else:
                    value = value_type(value.strip())

                if write_to_subdict:
                    sub_dict[key] = value
                else:
                    dictionary[key] = value

    return dictionary


def load_experiment(name, options_to_overwrite=None):
    experiment_path = os.path.join("..", "experimental_setups", name + ".txt")  # TODO: make this an optional input argument

    experiment = _file2dict(experiment_path)

    config_path = os.path.join("..", "experimental_setups", "configs", experiment.config_file + ".txt")
    experiment["config"] = _file2dict(config_path)
    # Assemble paths
    for key in experiment.config.keys():
        if key.endswith("_dir") and key != "base_dir":
            experiment.config[key] = os.path.join(experiment.config["base_dir"],
                                                  experiment.config[key],
                                                  experiment.config.dataset,
                                                  experiment.config.experiment_name) + os.sep

    for model_dict in experiment.models:
        opt_path = os.path.join("..", "experimental_setups", "opt_params", model_dict.opt_params_file + ".txt")
        model_dict["opt_params"] = _file2dict(opt_path)

        model_path = os.path.join("..", "experimental_setups", "model_params", model_dict.model_params_file + ".txt")
        model_dict["model_params"] = _file2dict(model_path)

    # Propensity model
    if "propensity_model" in experiment.keys():
        model_dict_prop = experiment.propensity_model[0]
        opt_path_prop = os.path.join("..", "experimental_setups", "opt_params", model_dict_prop.opt_params_file + ".txt")
        model_dict_prop["opt_params"] = _file2dict(opt_path_prop)
        model_path_prop = os.path.join("..", "experimental_setups", "model_params", model_dict_prop.model_params_file + ".txt")
        model_dict_prop["model_params"] = _file2dict(model_path_prop)

    if options_to_overwrite:
        for option in options_to_overwrite:
            key, value = option.split(":")
            keys = key.split(".")
            exp_option = experiment
            for key_id in range(len(keys)):
                if key_id == len(keys)-1:
                    exp_option[keys[key_id]] = type(exp_option[keys[key_id]])(value)
                else:
                    exp_option = exp_option[key]

    return experiment
