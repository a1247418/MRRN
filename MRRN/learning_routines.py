import os

import tensorflow as tf
from tensorflow.python.client import timeline
import tensorflow_probability as tfp
import sonnet as snt
import numpy as np

import utils
from loss import calculate_losses
import loss_def
from models.utils import str2model


def get_train_op(opt_params, loss, forbidden_name = None):
    '''
    Sets up an optimizer according to the given parameters and returns an operation to minimize the loss.
    :param opt_params: Parameter dictionary for the optimizer
    :param loss: Loss to minimize
    :param loss: A string. All gradient updates containing this string are ignored. (E.g. for "freezing" namespaces)
    :return: Optimization operation
    '''
    # Set up optimizer
    global_step = tf.train.get_or_create_global_step()
    lr = tf.train.exponential_decay(opt_params.learning_rate, global_step, 100, opt_params.lr_decay, staircase=True)

    if opt_params.optimizer is "rms":
        optimizer = tf.train.RMSPropOptimizer(lr, opt_params.rms_decay)
    elif opt_params.optimizer is "adagrad":
        optimizer = tf.train.AdagradOptimizer(lr)
    else:
        optimizer = tf.train.AdamOptimizer(lr)

    gradients = optimizer.compute_gradients(loss)

    # Disregarding gradients in disconnected (e.g. propensity) subgraph
    gradients = [(var, value) for var, value in gradients if var is not None]
    # Exclude specific variables by name
    if forbidden_name is not None:
        gradients = [g for g in gradients if forbidden_name not in g[1].name]
    if opt_params.gradient_clipping:
        gradients = [(tf.clip_by_value(gradient, -opt_params.gradient_magnitude, opt_params.gradient_magnitude),
                          value) for gradient, value in gradients]

    train_op = optimizer.apply_gradients(gradients)

    return train_op


def save_model(experiment, saver, model_name, run_id, session):
    save_dir = utils.assemble_model_path(experiment, model_name, run_id)
    print("Saving model to : " + save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    saver.save(session, save_dir, global_step=tf.train.get_or_create_global_step())


def _prepare_losses(experiment, model_instance, opt_params, meta, losses_to_calculate, tr_iter, val_iter, eval_iter=None):

    treatment_types = meta["n_treatments"]
    propensity_model_needed = any([loss_def.needs_propensity(loss) for loss in losses_to_calculate])
    matching_propensity_distrib = None
    losses = []

    for it in [tr_iter, val_iter, eval_iter]:
        is_training = it == tr_iter

        if it is None:
            losses.append(None)
            continue

        if propensity_model_needed:
            propensity_model_instance = str2model(experiment.propensity_model[0].model_type)(
                experiment.propensity_model[0].model_params, treatment_types)
            mu_hat, sig_hat = propensity_model_instance(it["x"][..., 0], False)
            propensity_distrib = tfp.distributions.MultivariateNormalDiag(loc=mu_hat, scale_diag=sig_hat)
            if it == tr_iter:
                matching_propensity_distrib = propensity_distrib
        else:
            propensity_distrib = None

        n_pcf_samples = meta['n_pcf_samples']

        outputs, normalized_rep = model_instance(it['x'], it['t'], it['s'], is_training, treatment_types)
        outputs_pcf = []
        for j in range(n_pcf_samples):
            curr_out_pcf, _ = model_instance(it['x'], it['t'], it['s_pcf'][:, :, j], is_training, treatment_types,
                                             skip_bin=True)
            outputs_pcf.append(curr_out_pcf)
        outputs_pcf = tf.stack(outputs_pcf, 2)

        curr_loss = calculate_losses(outputs, normalized_rep, it, meta, outputs_pcf,
                                        opt_params, propensity_distrib, tr_iter,
                                        matching_propensity_distrib, losses_to_calculate)
        losses.append(curr_loss)

    return losses[0], losses[1], losses[2]


def _train(experiment, model_setup, session, train_op, assigns, assigns_inputs, run_options, run_metadata,
           losses_objective, val_losses, losses_to_record):
    opt_params = model_setup.opt_params

    loss_records_val = {loss: [] for loss in losses_to_record}

    print("\n%s ------- TRAINING %s -------" % (utils.get_time_str(), model_setup.model_name))
    print("\t\t" + "\t".join([l for l in losses_to_record]))

    # Loss of last iteration block
    last_loss = np.finfo(np.float32).max
    # Whether early stopping criterion is met
    stop_training = False
    iteration_stopped = 0
    best_loss_id = -1
    # Number of iteration-blocks to look at after the first time there is no improvements
    early_stopping_lookahead_max = opt_params.early_stopping_lookahead
    early_stopping_lookahead = early_stopping_lookahead_max

    trainable_vars = tf.trainable_variables()

    for train_iter in range(opt_params.iterations):
        # Train
        if experiment.config.use_tracing:
            session.run(train_op, options=run_options, run_metadata=run_metadata)
        else:
            session.run(train_op)

        # Record losses every x iterations
        if train_iter % experiment.config.print_interval == 0 or train_iter == opt_params.iterations - 1:
            cur_val_losses = session.run([val_losses[l_name] for l_name in losses_to_record])
            cur_val_losses_dict = dict(zip(losses_to_record, cur_val_losses))

            to_print = [train_iter] + cur_val_losses
            print_form = "Iter%04d:" + "\t%.2f" * len(losses_to_record)
            print(print_form % tuple(to_print))

            # Store losses for later display
            for i, loss_name in enumerate(losses_to_record):
                loss_records_val[loss_name].append(cur_val_losses[i])

            # Early stopping
            early_stopping_loss = sum(cur_val_losses_dict[loss] for loss in losses_objective)
            # If stopping criterion is met - restore last weights
            if opt_params.early_stopping:
                if early_stopping_loss > last_loss:
                    # If look-ahead is used up: load best weights
                    if early_stopping_lookahead == 0:
                        print("Restoring best weights from iteration %d/%d due to early stopping." % (
                            iteration_stopped, opt_params.iterations))
                        session.run(assigns, dict(zip(assigns_inputs, trainable_vars_values)))
                        stop_training = True
                    else:
                        early_stopping_lookahead -= 1

                # Else: save current weights
                else:
                    last_loss = early_stopping_loss
                    trainable_vars_values = session.run(trainable_vars)
                    early_stopping_lookahead = early_stopping_lookahead_max
                    iteration_stopped = train_iter
                    best_loss_id = len(loss_records_val["MSE_F"]) - 1
                    print("Saving weights")

            if stop_training:
                break

    if not opt_params.early_stopping or not stop_training:
        best_loss_id = len(loss_records_val["MSE_F"]) - 1

    return loss_records_val, best_loss_id


def _evaluate(experiment, model_setup, session, eval_losses, losses_to_record):
    opt_params = model_setup.opt_params

    loss_records_eval = {loss: [] for loss in losses_to_record}

    print("\n%s ------- EVALUATING %s -------" % (utils.get_time_str(), model_setup.model_name))
    print("\t\t" + "\t".join([l for l in losses_to_record]))

    eval_iter = 0
    # Loops until the iterator ends
    while True:
        # Record losses every x iterations
        cur_val_losses = session.run([eval_losses[l_name] for l_name in losses_to_record])
        cur_val_losses_dict = dict(zip(losses_to_record, cur_val_losses))

        to_print = [eval_iter] + cur_val_losses
        print_form = "Iter%04d:" + "\t%.2f" * len(losses_to_record)
        print(print_form % tuple(to_print))

        # Store losses for later display
        for i, loss_name in enumerate(losses_to_record):
            loss_records_eval[loss_name].append(cur_val_losses[i])

        eval_iter +=1

    return loss_records_eval


def train_and_evaluate(experiment, model_setup, meta, run_id, tr_iter, val_iter, eval_iter=None, save=False, propensity_params=None):
    params = model_setup.params
    opt_params = model_setup.opt_params
    model_instance = str2model(model_setup.model_typ)(params, meta["n_treatments"])

    # Set up losses
    losses_objective = model_setup.opt_params.train_loss.split(",")
    losses_to_record = losses_objective + experiment.additional_losses_to_record.split(",")
    tr_losses, val_losses, eval_losses = _prepare_losses(experiment, model_instance, params, opt_params, meta,
                                                        losses_to_record, tr_iter, val_iter, eval_iter)

    # Set up optimizer & initialization
    objective_loss_train_tensor = tf.add_n([tr_losses[loss] for loss in losses_objective])
    train_op = get_train_op(opt_params, objective_loss_train_tensor, "multipleNN")
    init_op = [tf.global_variables_initializer(), tr_iter.initializer, val_iter.initializer]
    if eval_iter is not None:
        init_op += [eval_iter]

    # Making trainable variables assignable so that they can be restored in early stopping
    trainable_vars = tf.trainable_variables()
    assigns_inputs = [tf.placeholder(dtype=var.dtype, name="assign" + str(i)) for i, var in enumerate(trainable_vars)]
    assigns = [tf.assign(var, assigns_inputs[i]) for i, var in enumerate(trainable_vars)]

    # Load propensity model
    saver = snt.get_saver(model_instance)
    if propensity_params is not None:
        propensity_model_instance = str2model(experiment.propensity_model[0].model_type)(propensity_params, meta["n_treatments"])
        propensity_saver = snt.get_saver(propensity_model_instance)

    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    # Training loop
    with tf.train.MonitoredTrainingSession() as session:
        session.run(init_op)

        if propensity_params is not None:
            prop_path = utils.assemble_model_path(experiment, experiment.propensity_model.model_name, run_id)
            propensity_saver.restore(session._sess._sess._sess._sess, tf.train.latest_checkpoint(prop_path))

        loss_records_val, best_loss_id = _train(experiment, model_setup, session, train_op, assigns, assigns_inputs, run_options, run_metadata,
               losses_objective, val_losses, losses_to_record)

        if save:
            save_model(experiment, saver, model_setup.model_name, run_id, session._sess._sess._sess._sess)

        # Create timeline for performance analysis
        if experiment.config.use_tracing:
            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            with open('timeline.json', 'w') as f:
                f.write(ctf)

        if eval_iter is not None:
            loss_records_eval = _evaluate(experiment, model_setup, session, eval_losses, losses_to_record)

    return loss_records_val, best_loss_id, loss_records_eval
