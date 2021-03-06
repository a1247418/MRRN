import os
from enum import Enum

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import sonnet as snt

import utils
import models.utils
from matching_def import Group, Weighting, Space


def _pdist2sq(X, Y):
    """
    Computes the squared Euclidean distance between all pairs x in X, y in Y.
    Source: https://github.com/clinicalml/cfrnet/
    """
    C = -2 * np.matmul(X, np.transpose(Y))
    nx = np.sum(np.square(X), 1, keepdims=True)
    ny = np.sum(np.square(Y), 1, keepdims=True)
    D = (C + np.transpose(ny)) + nx
    return D


def _pdist2(X, Y):
    """
    Returns the tensorflow pairwise distance matrix.
    Source: https://github.com/clinicalml/cfrnet/
    """
    return np.sqrt(np.clip(_pdist2sq(X, Y), 1e-10, np.inf))


def _save_matching(experiment, matching_setup, run_id, matching_results):
    """ Saves the given matching results to a file. """
    path = utils.assemble_matching_path(experiment, matching_setup.to_string(), run_id)
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(path + "matching", matching_results)


def produce_grouping(t, s, treatment_types, levels=1):
    """
    Preprocessing to matching: Groups members of different treatments or treatment-subranges into lists.
    Assumes s to be in range [0,1]
    :param t: Treatment vectors of dimension n_samples x n_treatments
    :param s: Treatment strength vectors of dimension n_samples x n_treatments
    :param treatment_types: list of treatment types. 0: binary, 1: continuous
    :param levels: Into how many parts the treatment range should be partitioned.
    :return: Vector of group indicator. Lists of groupings: n_groups x n_grouping_members
    """
    assert levels >= 1

    n_treatments = len(s[0])

    group_list = []
    group_definitions = []
    for treatment in range(n_treatments):
        for level in range(levels):
            if treatment_types[treatment] == 1 or level == 0:
                group_list.append(np.array([]))
                group_id = len(group_list)-1

                s_t = s[t == treatment, treatment]
                idx_s_t = np.arange(len(t))[t==treatment]

                lower_bound = np.percentile(s_t, 100*level/levels) * (level > 0)
                upper_bound = np.percentile(s_t, 100*(level+1)/levels)
                group_definitions.append(Group(treatment, lower_bound, upper_bound))

                for i in range(len(s_t)):
                    if lower_bound < s_t[i] <= upper_bound:
                        group_list[group_id] = np.append(group_list[group_id], idx_s_t[i])

                group_list[group_id] = group_list[group_id].astype(int)

    return group_list, group_definitions


def produce_matching(experiment, it_to_match, it_matches, treatment_types, matching_setup, run_id,
                     save=False):
    """
    Calculates the matching-based estimation of y, given the data and a matching specification.
    :param experiment: experiment specification
    :param it_to_match: iterator of the data to estimate y
    :param it_matches: data to find matches from. if None, it_to_match will be used instead
    :param treatment_types: list of treatment types
    :param matching_setup: matching procedure specification of type MatchingSetup
    :param propensity_model_instance: optional propensity model instance
    :return: estimates of y_hat for data to match, group definitions
    """
    space = matching_setup.matching_space
    weighting = matching_setup.weighting
    k_max = matching_setup.k_max
    levels = matching_setup.levels
    needs_propensity_model = matching_setup.needs_propensity_model

    if needs_propensity_model:
        propensity_model_class = models.utils.str2model(experiment.propensity_model[0].model_type)
        propensity_model_instance = propensity_model_class(experiment.propensity_model[0].model_params,
                                                           len(treatment_types))

    session = tf.Session()
    init_op = [tf.global_variables_initializer(), it_to_match.initializer]
    if it_matches:
        init_op += [it_matches.initializer]
    session.run(init_op)

    if needs_propensity_model:

        mu_hat, sig_hat = propensity_model_instance(it_to_match.get_next()['x'], False)
        propensity_distrib = tfp.distributions.MultivariateNormalDiag(loc=mu_hat, scale_diag=sig_hat)

        if it_matches:
            mu_hat_m, sig_hat_m = propensity_model_instance(it_matches.get_next()['x'], False)
        else:
            mu_hat_m, sig_hat_m = propensity_model_instance(it_to_match.get_next()['x'], False)
        propensity_distrib_m = tfp.distributions.MultivariateNormalDiag(loc=mu_hat_m, scale_diag=sig_hat_m)

        # Restore propensity model parameters
        propensity_saver = snt.get_saver(propensity_model_instance)

        prop_path = utils.assemble_model_path(experiment, "prop", run_id)
        propensity_saver.restore(session, tf.train.latest_checkpoint(prop_path))

    data_to_match = session.run(it_to_match.get_next())
    x = data_to_match['x']
    z = x  # This is the space to weight on. currently it is always the space we match in.
    t = data_to_match['t']
    y = data_to_match['y']
    y = y[range(np.shape(y)[0]), t]
    s = data_to_match['s']
    if it_matches:
        data_matches = session.run(it_matches.get_next())
        x_m = data_matches['x']
        z_m = x_m
        t_m = data_matches['t']
        y_m = data_matches['y']
        y_m = y_m[range(np.shape(y_m)[0]), t_m]
        s_m = data_matches['s']
    else:
        x_m = None
        z_m = None
        t_m = t
        y_m = y
        s_m = s

    group_list_m, group_defs = produce_grouping(t_m, s_m, treatment_types, levels)
    n_groups = len(group_list_m)
    y_hat_all = np.tile(y[:, np.newaxis], [1, n_groups])

    for j in range(n_groups):
        i_to_match = np.array([i for i in range(np.shape(x)[0])]) if x_m is not None else \
            np.array([i for i in range(len(t_m)) if i not in group_list_m[j]])
        i_matches = group_list_m[j]
        len_to_match = len(i_to_match) if x_m is None else len(x)
        len_matches = len(i_matches)

        x_to_match = x[i_to_match] if x_m is None else x
        x_matches = x[i_matches] if x_m is None else x_m[i_matches]
        z_to_match = z[i_to_match] if z_m is None else z
        z_matches = z[i_matches] if z_m is None else z_m[i_matches]

        y_matches = y[i_matches] if x_m is None else y_m[i_matches]

        k = min(len_matches, k_max)

        # Find matches
        if space == Space.KL:
            assert propensity_model_instance is not None
            tfd = tfp.distributions

            m = propensity_distrib.mean()
            s = propensity_distrib.stddev()
            if x_m is None:
                m = tf.gather(m, i_to_match)
                s = tf.gather(s, i_to_match)
            m_tiled = tf.tile(m, [len_matches, 1])
            s_tiled = tf.tile(s, [len_matches, 1])

            m_m = propensity_distrib_m.mean()
            s_m = propensity_distrib_m.stddev()
            m_m = tf.gather(m_m, i_matches)
            s_m = tf.gather(s_m, i_matches)
            idx = []
            for i in range(len_matches):
                idx = idx + [i] * len_to_match
            m_rep = tf.gather(m_m, idx)
            s_rep = tf.gather(s_m, idx)

            # Symmetrized KL divergence
            D1 = tfd.MultivariateNormalDiag(loc=m_rep, scale_diag=s_rep).kl_divergence(
                tfd.MultivariateNormalDiag(loc=m_tiled, scale_diag=s_tiled))
            D2 = tfd.MultivariateNormalDiag(loc=m_tiled, scale_diag=s_tiled).kl_divergence(
                tfd.MultivariateNormalDiag(loc=m_rep, scale_diag=s_rep))
            D = tf.reshape(D1 + tf.transpose(D2), [len_to_match, len_matches])
            dist = session.run(D)

        elif space in [Space.PROPENSITY, Space.PROPENSITY_MEAN]:
            assert propensity_model_instance is not None
            m = propensity_distrib.mean()
            s = propensity_distrib.stddev()
            if x_m is None:
                m = tf.gather(m, i_to_match)
                s = tf.gather(s, i_to_match)
            m_m = propensity_distrib_m.mean()
            s_m = propensity_distrib_m.stddev()
            m_m = tf.gather(m_m, i_matches)
            s_m = tf.gather(s_m, i_matches)

            if space == Space.PROPENSITY:
                vec_to_match = tf.concat([m, s], 1)
                vec_matching = tf.concat([m_m, s_m], 1)
            elif space == Space.PROPENSITY_MEAN:
                vec_matching = m_m
                vec_to_match = m
            vec_to_match_eval, vec_matching_eval = session.run([vec_to_match, vec_matching])
            dist = _pdist2(vec_to_match_eval, vec_matching_eval)

        elif space == Space.NONE:
            dist = np.random.rand(len_to_match, len_matches)

        else:
            # Euclidean distance
            dist = _pdist2(x_to_match, x_matches)  # len_i x len_j
        k_smallest_idx = np.argsort(dist)[:, :k]  # indices within rows: len_i x k

        # Weight matches
        if weighting == Weighting.NONE:
            W = np.ones_like(dist)
        elif weighting == Weighting.EUCLIDEAN:
            if z is not None:
                dist_w = _pdist2(z_to_match, z_matches)
                W = 1 / (dist_w + (
                        dist_w == 0) + 1e-10)  # Count 0 dist samples as weight 1. this likely only occurs for binary s.
            else:
                W = 1 / (dist + (dist == 0) + 1e-10)
        else:
            print("WARNING: Unknown weighting for matching. Defaulting to Weighting.NONE")
            W = np.ones_like(dist)

        W_k = W[[[l] for l in np.arange(W.shape[0])], k_smallest_idx]  # len_i x k
        y_hat = np.sum(y_matches[k_smallest_idx] * W_k, 1) / np.sum(W_k, 1)

        if x_m is None:
            y_hat_all[i_to_match, j] = y_hat
        else:
            y_hat_all[:, j] = y_hat

    if save:
        matching_results = {
            "y_hat": y_hat_all,
            "groups": group_defs
        }
        _save_matching(experiment, matching_setup, run_id, matching_results)

    return y_hat_all, group_defs