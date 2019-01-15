import os

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import sonnet as snt


def _get_factuals_mask(t, n_treatments):
    stacking_helper = []
    for i in range(n_treatments):
        stacking_helper.append(tf.equal(t, i)[:, 0])
    idx_f = tf.stack(stacking_helper, axis=1)

    return idx_f


def _propensity_loss(s, mu_hat, sig_hat):
    """Returns the negative log probabilities of the factual tratment s, given a multivariate
    normal with mu_hat and covariance diagonal sig_hat."""

    normal = tfp.distributions.MultivariateNormalDiag(loc=mu_hat, scale_diag=sig_hat)

    log_probabilities = normal.log_prob(s)

    # Remove outliers
    mean, stdev = tf.nn.moments(log_probabilities, axes=[0])
    mask_lower = tf.greater(log_probabilities, mean - 3 * stdev)
    mask_upper = tf.less(log_probabilities, mean + 3 * stdev)
    mask = tf.logical_and(mask_lower, mask_upper)
    log_probabilities = tf.boolean_mask(log_probabilities, mask)

    neg_log_probability = -tf.reduce_mean(log_probabilities)
    return neg_log_probability, log_probabilities


def _regularization_loss():
    graph_regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    regularization_loss = tf.cast(tf.reduce_sum(graph_regularizers), tf.float32)
    return regularization_loss


def _prepare_data(tr_data, val_data, n_treatments, batch_size, is_train, run_id):
    # Make iterators from data
    if config.data_from_database:
        tr_iter_origin, val_iter_origin = data_loader.get_iterator_from_database(
            [1-config.validation_ratio, config.validation_ratio], [batch_size, None], [True, False], is_train, run_id)
        tr_iter_names = []
        val_iter_names = []
    else:
        # Throw away all columns that are not needed
        val_data_small = {}
        tr_data_small = {}
        for key in ["x", "t", "s"]:
            val_data_small[key] = tr_data[key]
            tr_data_small[key] = val_data[key]

        tr_iter_origin, tr_iter_names = data_loader.get_iterable_iterator_from_data(tr_data_small, batch_size=batch_size, shuffle=True)
        val_iter_origin, val_iter_names = data_loader.get_iterable_iterator_from_data(val_data_small, batch_size=val_data["n_samples"])

    tr_iter = tr_iter_origin.get_next()
    val_iter = val_iter_origin.get_next()

    # Prepare data
    x_tr = tr_iter["x"][..., 0]
    x_val = val_iter["x"][..., 0]
    t_tr = tr_iter["t"]
    t_val = val_iter["t"]
    s_tr = tr_iter["s"][..., 0]
    s_val = val_iter["s"][..., 0]

    # Only keep the factual s. Replace the rest with 0
    mask_tr_f = _get_factuals_mask(t_tr, n_treatments)
    mask_val_f = _get_factuals_mask(t_val, n_treatments)
    s_tr = tf.multiply(s_tr, tf.cast(mask_tr_f, tf.float32))
    s_val = tf.multiply(s_val, tf.cast(mask_val_f, tf.float32))

    return x_tr, x_val, s_tr, s_val, t_tr, t_val, tr_iter_names, val_iter_names, tr_iter_origin, val_iter_origin


def get_propensity_model_class():
    if config.data_from_database:
        return ConvWrapper
    else:
        return multipleNN


def get_objective_list():
    return [Loss.MSE_F.name, Loss.REG.name]


# TODO: account for the case of training the final model, when val_data is empty
# TODO: unify with training.train
def train(model_instance, params, opt_params, model_name, model_config_string,
          tr_data, val_data, meta, objective_loss_train, propensity_model_instance=None, additional_losses_to_record=[],
          save=False, use_train_data=True, run_id=0):
    n_treatments = meta["n_treatments"]

    print("Propensity Configuration: %s" % utils.params_to_string(params))
    print("%s" % utils.params_to_string(opt_params))

    x_tr, x_val, s_tr_pad, s_val_pad, t_tr, t_val, tr_iter_names, val_iter_names, tr_iter_origin, val_iter_origin = _prepare_data(tr_data, val_data, n_treatments, opt_params.batch_size, use_train_data, run_id)

    # Create the model
    mu_hat_tr, sig_hat_tr = model_instance(x_tr, True)
    mu_hat_val, sig_hat_val = model_instance(x_val, False)

    # Define losses
    objective_loss_train = utils.str2loss_list(params.train_loss)
    tr_loss_p, prob_tr = _propensity_loss(s_tr_pad, mu_hat_tr, sig_hat_tr)
    val_loss_p, prob_val = _propensity_loss(s_val_pad, mu_hat_val, sig_hat_val)

    tr_loss = tr_loss_p + _regularization_loss()

    # Define operations
    train_op = tf_utils.get_train_op(opt_params, tr_loss)
    init_op = [tf.global_variables_initializer(), val_iter_origin.initializer, tr_iter_origin.initializer]

    # Making trainable variables assignable so that they can be restored in early stopping
    trainable_vars = tf.trainable_variables()
    assigns_inputs = [tf.placeholder(dtype=var.dtype, name="assign" + str(i)) for i, var in
                      enumerate(trainable_vars)]
    assigns = [tf.assign(var, assigns_inputs[i]) for i, var in enumerate(trainable_vars)]

    best_loss = np.finfo(np.float32).max
    best_loss_id = 0
    loss_records_tr = {Loss.MSE_F.name: []}
    loss_records_val = {Loss.MSE_F.name: []}

    saver = snt.get_saver(model_instance)
    weights_stored = False

    # Training loop
    with tf.Session() as session:
        if config.data_from_database:
            session.run(init_op)
        else:
            feed_dicts = {tr_iter_names[k]: tr_data[k] for k in tr_iter_names.keys()}
            feed_dicts.update({val_iter_names[k]: val_data[k] for k in tr_iter_names.keys()})
            session.run(init_op, feed_dict=feed_dicts)

        for train_iter in range(opt_params.iterations):
            # Train
            session.run(train_op)

            # Record losses every x iterations
            if (train_iter > 15 and train_iter % config.print_interval_prop == 0) or train_iter == opt_params.iterations - 1:
                curr_tr_loss = session.run(tr_loss_p)
                curr_val_loss = session.run(val_loss_p)

                loss_records_tr[Loss.MSE_F.name].append(curr_tr_loss)
                loss_records_val[Loss.MSE_F.name].append(curr_val_loss)

                print("Iter%04d:\tPropensity loss: %.3f\t%.3f" % (train_iter, curr_tr_loss, curr_val_loss))

                # Break if loss takes on illegal value
                if np.isnan(curr_val_loss) or np.isnan(curr_tr_loss):
                    print("Illegal loss value. Aborting training.")
                    break

                # If loss improved: save weights, else: restore weights
                curr_loss = sum([loss_records_val[loss.name][len(loss_records_val[Loss.MSE_F.name])-1] for loss in objective_loss_train])
                if best_loss > curr_loss:
                    best_loss = curr_loss
                    best_loss_id = len(loss_records_val[Loss.MSE_F.name]) - 1
                    trainable_vars_values = session.run(trainable_vars)
                    weights_stored = True
                    print("Saving weights")

        # Restore variables of the best iteration
        if weights_stored:
            session.run(assigns, dict(zip(assigns_inputs, trainable_vars_values)))

        if save:
            name = model_name
            name += str(run_id + (not use_train_data)) if config.data_from_database else ""
            tf_utils.save_model(saver, name, session)

    return loss_records_tr, loss_records_val, best_loss_id


def train_from_default(tr_data, val_data, meta_data, is_train=True, run_id=0):
    # this assumes an already optimized params set
    params = utils.copy_dict_to_hparams(config.propensity_params_default)
    prop_path = config.saves_dir + "prop"+str(run_id + (not is_train)) + os.sep
    if tf.train.latest_checkpoint(prop_path):
        print("Found propensity model... skipping training.")
    else:
        print("Did not find propensity model in: %s. Training new one." % prop_path)
        opt_params = utils.copy_dict_to_hparams(config.optimization_params_prop_default)
        model_instance = get_propensity_model_class()(params, meta_data["n_treatments"])
        train(model_instance, params, opt_params, "prop", "prop" + utils.params_to_string(params),
                   tr_data, val_data, meta_data, get_objective_list(), propensity_model_instance=None, save=True,
                   use_train_data=is_train, run_id=run_id)

    # Cannot return a model instance, as one sonnet module can only be connected to one graph
    return params


def hparam_search(n_configs=50):
    print("Hyper parameter search: Propensity")
    params_dict = config.propensity_params
    opt_params_dict = config.optimization_params_prop
    model_name = "prop"
    training_func = train
    return tf_utils.hparam_search(multipleNN, params_dict, opt_params_dict, model_name, training_func,
                                  n_configs=n_configs, hparam_losses_to_record=[])
