import tensorflow as tf
import numpy as np
import utils
from loss_def import RegisteredLosses
from matching_def import MatchingSetup, matching_from_string


def _pdist2sq(X, Y):
    """
    Computes the squared Euclidean distance between all pairs x in X, y in Y.
    Source: https://github.com/clinicalml/cfrnet/
    """
    C = -2 * tf.matmul(X, tf.transpose(Y))
    nx = tf.reduce_sum(tf.square(X), 1, keepdims=True)
    ny = tf.reduce_sum(tf.square(Y), 1, keepdims=True)
    D = (C + tf.transpose(ny)) + nx
    return D


def _pdist2(X, Y):
    """
    Returns the tensorflow pairwise distance matrix
    Source: https://github.com/clinicalml/cfrnet/
    """
    return utils.safe_sqrt(_pdist2sq(X, Y))


def _get_factuals_mask(t, n_treatments):
    stacking_helper = []
    for i in range(n_treatments):
        stacking_helper.append(tf.equal(t, i)[:, 0])
    idx_f = tf.stack(stacking_helper, axis=1)

    return idx_f


def _get_all_counterfactuals(dat, dat_pcf, n_treatments, treatment_types_all, n_pfc_samples):
    """ Returns a matrix of shape [batch_size x n_param_treatments x n_pcf_samples],
    for counterfactuals, by combining counterfactual data items of binary and parametric origin. """
    dat_cf_all = []
    for i in range(n_pfc_samples):
        dat_cf_curr = []

        t_cf_count = 0
        for j in range(n_treatments):
            if treatment_types_all[j]:
                dat_cf_curr.append(dat_pcf[:, t_cf_count, i])
                t_cf_count += 1
            else:
                dat_cf_curr.append(dat[:, j])
        dat_cf_all.append(tf.stack(dat_cf_curr, axis=1))  # batch_size x n_treatments
    dat_cf_all = tf.stack(dat_cf_all, axis=2)  # batch_size x n_treatments x n_cf_samples

    return dat_cf_all


def _calc_cf_error_from_full_matrix(y, out, t, sample_weight, meta):
    """
    Calculates the counterfactual error from a n_samples x n_treatments tensor, containing only cf predictions.
    :param y: True outcome
    :param out: Predicted outcome
    :param t: Treatment assignment
    :param sample_weight: n_samples long vector of wighting of the error for each sample
    :param meta: Meta data
    :return: MSE(y, out)
    """

    n_treatments = meta["n_treatments"]
    treatment_types = meta["treatment_types"]
    treatment_types_all = [i for i in ([0] + treatment_types)]
    neg_treatment_types_all = [not i for i in ([0] + treatment_types)]  # add binary treatment as control
    n_param_treatments = sum(treatment_types)
    n_bin_treatments = n_treatments - n_param_treatments
    n_samples = tf.shape(t)[0]

    mask_cf = tf.logical_not(_get_factuals_mask(t, n_treatments))
    mask_bcf = tf.logical_and(mask_cf, tf.tile(np.array(neg_treatment_types_all)[np.newaxis, :], [n_samples, 1]))
    mask_p = tf.tile(np.array(treatment_types_all)[np.newaxis, :], [n_samples, 1])

    # Bin cf error
    y_bcf = tf.boolean_mask(y, mask_bcf)
    out_bcf = tf.boolean_mask(out, mask_bcf)
    sample_weight_b = tf.boolean_mask(tf.tile(sample_weight, [1, n_treatments]), mask_bcf)
    sqr_loss_bcf_list = tf.reshape(sample_weight_b * tf.squared_difference(out_bcf, y_bcf), [-1, 1])
    sqr_loss_bcf = (n_bin_treatments/n_treatments) * tf.reduce_mean(sqr_loss_bcf_list)
    loss_bcf = (n_bin_treatments/n_treatments) * tf.reduce_mean(tf.reshape(tf.abs(sample_weight_b * (out_bcf - y_bcf)), [-1, 1]))

    # Param error
    y_p = tf.boolean_mask(y, mask_p)
    out_p = tf.boolean_mask(out, mask_p)
    sample_weight_p = tf.boolean_mask(tf.tile(sample_weight, [1, n_treatments]), mask_p)
    sqr_loss_pcf_list = tf.reshape(sample_weight_p * tf.squared_difference(out_p, y_p), [-1, 1])
    sqr_loss_pcf = (n_param_treatments/n_treatments) * tf.reduce_mean(sqr_loss_pcf_list)
    loss_pcf = (n_param_treatments/n_treatments) * tf.reduce_mean(tf.reshape(tf.abs(sample_weight_p * (out_p - y_p)), [-1, 1]))

    # Calculate std deviation
    seq_bcf = tf.reshape(tf.abs(sample_weight_b * (out_bcf - y_bcf)), [-1, 1])
    mean_bcf = tf.reduce_mean(seq_bcf, 0)
    pre_var_bcf = tf.reduce_sum(n_bin_treatments * tf.pow(seq_bcf-mean_bcf, 2.), 0)[0]

    seq_p = tf.reshape(tf.abs(sample_weight_p * (out_p - y_p)), [-1, 1])
    mean_p = tf.reduce_mean(seq_p, 0)
    pre_var_p = tf.reduce_sum(n_param_treatments * tf.pow(seq_p-mean_p, 2.), 0)[0]

    std = (pre_var_bcf+pre_var_p)/(n_param_treatments * tf.cast(tf.shape(out_p), dtype=tf.float32) +
                                  (n_bin_treatments * tf.cast(tf.shape(out_bcf), dtype=tf.float32)))
    std_loss_tot = tf.sqrt(std)

    # Calculate std deviation for squared loss
    pre_var_bcf_sqr = tf.reduce_sum(n_bin_treatments * tf.pow(sqr_loss_bcf_list-sqr_loss_bcf, 2.), 0)[0]
    pre_var_p_sqr = tf.reduce_sum(n_param_treatments * tf.pow(sqr_loss_pcf_list-sqr_loss_pcf, 2.), 0)[0]

    std_sqr = (pre_var_bcf_sqr+pre_var_p_sqr)/(n_param_treatments * tf.cast(tf.shape(out_p), dtype=tf.float32) +
                                  (n_bin_treatments * tf.cast(tf.shape(out_bcf), dtype=tf.float32)))
    std_loss_tot_sqr = tf.sqrt(std_sqr)

    sqr_loss_tot = sqr_loss_bcf + sqr_loss_pcf
    loss_tot = loss_bcf + loss_pcf
    return loss_tot, sqr_loss_tot, std_loss_tot, std_loss_tot_sqr


def _pre_matching_losses(data, meta, s_cf_all, outputs_cf_all, sample_weight, n_pfc_samples, loss_list):
    n_treatments = meta["n_treatments"]

    m_losses_dict = {}
    for loss_name in loss_list:
        losses_sqr_temp = []
        losses_root_temp = []
        losses_std_temp = []
        losses_std_sqr_temp = []

        # If the required matching data is not available: skip
        if ("groups_"+loss_name) not in meta:
            m_losses_dict[loss_name] = tf.constant(0.0)
            continue

        group_defs = meta["groups_" + loss_name]
        matching_column_name = "y_" + loss_name

        for i in range(n_pfc_samples):
            group_assignments = _s2group(s_cf_all[:, :, i], group_defs, n_treatments)  # bs x n_treatments
            y_m = data[matching_column_name]  # bs x n_groups
            y_m_inplace = tf.zeros_like(group_assignments)  # bs x n_treatments

            for g_id in range(len(group_defs)):
                where_g_id = tf.cast(tf.equal(group_assignments, g_id), dtype=tf.float32)  # bs x n_treatments
                y_m_inplace += tf.multiply(where_g_id, tf.tile(y_m[:, g_id, tf.newaxis], [1, n_treatments]))

            loss_root, loss_sqr, loss_std, loss_std_sqr = _calc_cf_error_from_full_matrix(y_m_inplace, outputs_cf_all[:, :, i], data["t"], sample_weight, meta)
            losses_sqr_temp.append(loss_sqr)
            losses_root_temp.append(loss_root)
            losses_std_temp.append(loss_std)
            losses_std_sqr_temp.append(loss_std_sqr)

        m_losses_dict[loss_name] = tf.reduce_mean(losses_sqr_temp)# MSE
        m_losses_dict[loss_name+"_root"] = tf.reduce_mean(losses_root_temp)# Mean Error
        m_losses_dict[loss_name+"_std"] = tf.reduce_mean(losses_std_temp)# Stdev Mean Error
        m_losses_dict[loss_name+"_std_sqr"] = tf.reduce_mean(losses_std_sqr_temp)# Stdev MSE

    return m_losses_dict


def _s2group(s_cf, group_defs, n_treatments):
    """Converts a matrix of counterfactual treatment assignments to matching group IDs."""
    group_assignments = []
    for t_id in range(n_treatments):
        assigment_vec = tf.zeros_like(s_cf[:, 0])  # bs x ,
        for i, gd in enumerate(group_defs):
            if t_id == gd.t:
                l1 = tf.greater(s_cf[:, gd.t], gd.lower)
                u1 = tf.less(s_cf[:, gd.t], gd.upper)
                u2 = tf.equal(s_cf[:, gd.t], gd.upper)
                assigment_vec += tf.cast(tf.logical_and(l1, tf.logical_or(u1, u2)), dtype=tf.float32) * i
        group_assignments.append(assigment_vec)
    return tf.stack(group_assignments, axis=1)  # bs x n_treatments


def nn_approx(data, data_m, s_pcf, s_pcf_m, n_treatments, k_max, rem_outliers, reweight=None, space="x",
              propensity_distrib=None, propensity_distrib_m=None):
    """
    Approximates counterfactual outcomes by the average of k nearest neighbours.
    :param k: Number of nearest neighbours to average
    :return: A tensor of shape [batch_size,n_treatments], or [batch_size,n_treatments,n_pcf_samples], containing
    factual outcomes and counterfactual approximations.
    """
    x = data["x"][:, :, 0]
    t = data["t"]
    mask_f = _get_factuals_mask(t, n_treatments)
    y_f = tf.boolean_mask(data["y"], mask_f)
    s_f = tf.multiply(data["s"][:, :, 0], tf.cast(mask_f, dtype=tf.float32))

    if data_m is None:
        x_m = x
        t_m = t
        s_f_m = s_f
        y_f_m = y_f
    else:
        assert s_pcf_m is not None
        x_m = data_m["x"][:, :, 0]
        t_m = data_m["t"]
        mask_f_m = _get_factuals_mask(t_m, n_treatments)
        y_f_m = tf.boolean_mask(data_m["y"], mask_f_m)
        s_f_m = tf.multiply(data_m["s"][:, :, 0], tf.cast(mask_f_m, dtype=tf.float32))

    do_sort = reweight is "rank_lin" or reweight is "rank_exp"
    y_nn = []
    y_nn_all = []
    smallest_k_all = []

    for t_id in range(n_treatments):
        # i0: indices to match (not having t = t_id)
        # i1: indices to match on (t = t_id)
        i1 = tf.to_int32(tf.where(tf.equal(t_m, t_id))[:, 0])  # <=bs_m
        if data_m is None:
            i0 = tf.to_int32(tf.where(tf.not_equal(t, t_id))[:, 0])  # <=bs
        else:
            i0 = tf.to_int32(tf.where(tf.ones_like(t, dtype=tf.bool))[:, 0])

        # Reducing k to the number of samples with t = t_id, if necessary.
        k = tf.minimum(k_max, tf.shape(i1)[0])

        x1 = tf.gather(x_m, i1)  # len_i1 x dim_x
        x0 = tf.gather(x, i0)  # len_0 x dim_x

        # Set all columns for s besides the t_id to 0
        s_pad = s_pcf[:, t_id, tf.newaxis]  # eventually: bs x n_treatments
        if t_id > 0:
            leading_zeros = tf.reshape(tf.zeros_like(s_pcf)[:, :t_id], [-1, t_id])
            s_pad = tf.concat([leading_zeros, s_pad], axis=1)
        if t_id < n_treatments - 1:
            trailing_zeros = tf.reshape(tf.zeros_like(s_pcf)[:, :(n_treatments - 1 - t_id)],
                                        [-1, n_treatments - 1 - t_id])
            s_pad = tf.concat([s_pad, trailing_zeros], axis=1)

        # All methods requiring a propensity model
        if space.startswith("prop"):
            assert propensity_distrib is not None

            if data_m is None:
                if space is "prop_point":
                    propensities = propensity_distrib.log_prob(s_pad)
                    prop0 = tf.gather(propensities, i0)[:, tf.newaxis]
                    prop1 = tf.gather(propensities, i1)[:, tf.newaxis]
                else:
                    if space is "prop":
                        prop_params = tf.concat([propensity_distrib.mean(), propensity_distrib.stddev()], axis=1)
                    elif space is "prop_mean":
                        prop_params = propensity_distrib.mean()
                    prop0 = tf.gather(prop_params, i0)
                    prop1 = tf.gather(prop_params, i1)

            else:
                assert propensity_distrib_m is not None
                s_pad_m = s_pcf_m[:, t_id, tf.newaxis]  # eventually: bs x n_treatments
                if t_id > 0:
                    leading_zeros = tf.reshape(tf.zeros_like(s_pcf_m)[:, :t_id], [-1, t_id])
                    s_pad_m = tf.concat([leading_zeros, s_pad_m], axis=1)
                if t_id < n_treatments-1:
                    trailing_zeros = tf.reshape(tf.zeros_like(s_pcf_m)[:, :(n_treatments-1-t_id)], [-1, n_treatments-1-t_id])
                    s_pad_m = tf.concat([s_pad_m, trailing_zeros], axis=1)

                if space == "prop_point":
                    propensities0 = propensity_distrib.log_prob(s_pad)
                    prop0 = tf.gather(propensities0, i0)[:, tf.newaxis]
                    propensities1 = propensity_distrib_m.log_prob(s_pad_m)
                    prop1 = tf.gather(propensities1, i1)[:, tf.newaxis]
                else:
                    if space is "prop":
                        prop_params0 = tf.concat([propensity_distrib.mean(), propensity_distrib.stddev()], axis=1)
                        prop_params1 = tf.concat([propensity_distrib_m.mean(), propensity_distrib_m.stddev()], axis=1)
                    elif space is "prop_mean":
                        prop_params0 = propensity_distrib.mean()
                        prop_params1 = propensity_distrib_m.mean()
                    prop0 = tf.gather(prop_params0, i0)
                    prop1 = tf.gather(prop_params1, i1)
            D = _pdist2(prop0, prop1)
        elif space is "rand":
            D = tf.random_uniform([tf.shape(i0)[0], tf.shape(i1)[0]])
        elif space is "s":
            D = _pdist2(tf.gather(s_pcf, i0), tf.gather(s_f_m, i1))
        elif space is "x":
            D = _pdist2(x0, x1)
        elif space is not None:
            print("WARNING: Matching space not found. "+space)
            D = _pdist2(x0, x1)  # len_i0 x len_i1

        smallest_k, i_smallest_k = tf.nn.top_k(-D, k, sorted=do_sort)  # len_i0 x k
        smallest_k = -smallest_k
        smallest_k_all.append(smallest_k)

        nn_indices = tf.gather(i1, i_smallest_k)
        y_nn_t_concat = tf.gather(y_f_m[:, 0], nn_indices)  # len_i0 x k
        y_nn_all.append(y_nn_t_concat)

        if reweight is "rank_lin":
            def k_greater_0():
                weight_row = tf.linspace(1.0, 0.1, k)[tf.newaxis, :]
                weights = tf.tile(weight_row, [tf.shape(i0)[0], 1])
                return weights
            def k_is_0():
                return tf.zeros([0, 0], tf.float32)
            weights = tf.cond(tf.equal(k, 0), k_is_0, k_greater_0)
        elif reweight is "rank_exp":
            def k_greater_0():
                weight_row = tf.pow(tf.linspace(10.0, 1.0, k), 2.0)[tf.newaxis, :]
                weights = tf.tile(weight_row, [tf.shape(i0)[0], 1])
                return weights
            def k_is_0():
                return tf.zeros([0, 0], tf.float32)
            weights = tf.cond(tf.equal(k, 0), k_is_0, k_greater_0)
        elif reweight is "dist":
            weights = tf.ones_like(smallest_k)/(smallest_k + tf.cast(tf.equal(smallest_k, 0.), dtype=tf.float32) + 1e-10)
        elif reweight is "str":
            s_dist = _pdist2(tf.gather(s_pcf, i0), tf.gather(s_f_m, i1))
            smallest_s_dist = tf.one_hot(i_smallest_k, tf.shape(i1)[0], on_value=1.0, off_value=0.0, axis=-1)
            smallest_s_dist = tf.reduce_sum(smallest_s_dist, 1)
            smallest_s_dist = tf.reshape(smallest_s_dist, [-1, tf.shape(i1)[0]])
            smallest_s_dist = tf.boolean_mask(s_dist, smallest_s_dist)
            smallest_s_dist = tf.reshape(smallest_s_dist, tf.shape(i_smallest_k))
            weights = tf.ones_like(smallest_s_dist) / (smallest_s_dist + tf.cast(tf.equal(smallest_s_dist, 0.), dtype=tf.float32) + 1e-10)
        elif reweight is not None:
            print("WARNING: Matching weighting not found. " + reweight)
        else:
            weights = tf.ones(shape=[tf.shape(i0)[0], k])

        # Filter out outliers
        if rem_outliers:
            mean_nn_y, std_nn_y = tf.nn.moments(y_nn_t_concat, axes=[1])
            mean_nn_y = mean_nn_y[:, tf.newaxis]
            std_nn_y = std_nn_y[:, tf.newaxis]
            outlier_selector_L = tf.greater(y_nn_t_concat, mean_nn_y-3*std_nn_y)  # len_i0 x k
            outlier_selector_U = tf.less(y_nn_t_concat, mean_nn_y+3*std_nn_y)  # len_i0 x k
            outlier_selector = tf.cast(tf.logical_and(outlier_selector_L, outlier_selector_U), dtype=tf.float32)
        else:
            outlier_selector = tf.ones(shape=[tf.shape(i0)[0], k])

        # Reweight sum of y
        weighted_outlier_selector = tf.multiply(outlier_selector, weights)  # len_i0 x k
        y_nn_t_selected = tf.multiply(y_nn_t_concat, weighted_outlier_selector)  # len_i0 x k
        y_nn_t_sum = tf.reduce_sum(y_nn_t_selected, 1) / tf.reduce_sum(weighted_outlier_selector, 1)  # len_i0

        y_nn_t_sum = y_nn_t_sum[:, tf.newaxis]
        if data_m is None:
            y_nn.append(tf.dynamic_stitch([i0, i1], [y_nn_t_sum, tf.gather(y_f, i1)]))  # bs
        else:
            y_nn.append(y_nn_t_sum)

    y_nn = tf.stack(y_nn, 1)  # bs x n_t
    return y_nn, y_nn_all, smallest_k_all


def wasserstein_loss(X, t, p, t_id, lam=10, its=10, sq=False):
    """
    Returns the Wasserstein distance between treatment groups
    Source: https://github.com/clinicalml/cfrnet/
    """
    ic = tf.where(tf.equal(t, 0))[:, 0]
    it = tf.where(tf.equal(t, t_id))[:, 0]
    Xc = tf.gather(X, ic)
    Xt = tf.gather(X, it)
    nc = tf.to_float(tf.shape(Xc)[0])
    nt = tf.to_float(tf.shape(Xt)[0])

    ''' Compute distance matrix'''
    if sq:
        M = _pdist2(Xt, Xc)
    else:
        M = utils.safe_sqrt(_pdist2(Xt, Xc))
    M = tf.cast(M, tf.float32)

    ''' Estimate lambda and delta '''
    M_mean = tf.reduce_mean(M)
    M_drop = tf.nn.dropout(M, 10.0 / (nc * nt))
    delta = tf.stop_gradient(tf.reduce_max(M))
    eff_lam = tf.stop_gradient(lam / M_mean)

    ''' Compute new distance matrix '''
    Mt = M
    row = delta * tf.ones(tf.shape(M[0:1, :]))
    col = tf.concat([delta * tf.ones(tf.shape(M[:, 0:1])), tf.zeros((1, 1))], 0)
    Mt = tf.concat([M, row], 0)
    Mt = tf.concat([Mt, col], 1)

    ''' Compute marginal vectors '''
    a = tf.concat([p * tf.ones(tf.shape(tf.where(tf.equal(t, t_id))[:, 0:1])) / nt, (1 - p) * tf.ones((1, 1))], 0)
    b = tf.concat([(1 - p) * tf.ones(tf.shape(tf.where(tf.equal(t, 0))[:, 0:1])) / nc, p * tf.ones((1, 1))], 0)

    ''' Compute kernel matrix'''
    Mlam = eff_lam * Mt
    K = tf.exp(-Mlam) + 1e-6  # added constant to avoid nan
    U = K * Mt
    ainvK = K / a

    u = a
    for i in range(0, its):
        u = 1.0 / (tf.matmul(ainvK, (b / tf.transpose(tf.matmul(tf.transpose(u), K)))))
    v = b / (tf.transpose(tf.matmul(tf.transpose(u), K)))

    T = u * (tf.transpose(v) * K)

    E = T * Mt
    D = 2 * tf.reduce_sum(E)

    return D


def regularization_loss():
    graph_regularizers = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    regularization_loss = tf.cast(tf.reduce_sum(graph_regularizers), tf.float32)
    return regularization_loss


def calculate_losses(outputs, normalized_rep, data, meta, outputs_pcf, opt_params, matching_data, loss_list):
    y = data['y'] # batch_size x n_treatments x 1
    t = data['t', tf.newaxis]  # batch_size x 1
    s = data['s'][...]  # batch_size x n_treatments x 1
    y_pcf = None if not 'y_pcf' in data.keys() else data['y_pcf']
    p_t = meta["p_t"]
    n_treatments = meta["n_treatments"]
    treatment_types = meta["treatment_types"]

    # Splitting counterfactual and factual samples
    mask_f = _get_factuals_mask(t, n_treatments)
    mask_cf = tf.logical_not(mask_f)

    # Sample reweighting
    sample_weight = tf.cast(tf.equal(t, 0), tf.float32) / 1  # TODO: p_t[0]
    for i in range(1, n_treatments):
        sample_weight += tf.cast(tf.equal(t, i), tf.float32) / 1  # TODO: p_t[i]

    # Factual Squared Loss
    y_f = tf.boolean_mask(y, mask_f)
    y_cf = tf.boolean_mask(y, mask_cf)
    outputs_f = tf.boolean_mask(outputs, mask_f)
    outputs_cf = tf.boolean_mask(outputs, mask_cf)
    sqr_loss_f = tf.reduce_mean(sample_weight * tf.squared_difference(outputs_f, y_f))

    # Counterfactual Squared Loss
    if y_pcf is None:
        sample_weight = tf.tile(sample_weight, [n_treatments-1, 1])
        sqr_loss_cf = tf.reduce_mean(tf.reshape(sample_weight * tf.squared_difference(outputs_cf, y_cf), [-1, 1]))
    else:
        # y_pcf has dimensions [sample_size, nr_parametric_treatments, nr_cf_samples, nr_simulations]
        treatment_types_all = [i for i in ([0] + treatment_types)]
        n_pfc_samples = y_pcf.shape[2]
        y_pcf = y_pcf
        s_pcf = data["s_pcf"]
        s_pcf_m = matching_data["s_pcf"]
        s_m = matching_data["s"]

        # Set up data shape for convenience
        y_cf_all = _get_all_counterfactuals(y, y_pcf, n_treatments, treatment_types_all, n_pfc_samples)
        s_cf_all = _get_all_counterfactuals(s, s_pcf, n_treatments, treatment_types_all, n_pfc_samples)
        outputs_cf_all = _get_all_counterfactuals(outputs, outputs_pcf, n_treatments, treatment_types_all, n_pfc_samples)

        sqr_loss_cf = []
        for i_pcf in range(n_pfc_samples):
            _, curr_sq_loss_cf, _, _ = _calc_cf_error_from_full_matrix(y_cf_all[:, :, i_pcf], outputs_cf_all[:, :, i_pcf], t,
                                                               sample_weight, meta)
            sqr_loss_cf.append(curr_sq_loss_cf)
        sqr_loss_cf = tf.reduce_mean(sqr_loss_cf)

        print("%s Setting up matching loss." % utils.get_time_str())
        pre_m_losses_dict = _pre_matching_losses(data, meta, s_cf_all, outputs_cf_all, sample_weight, n_pfc_samples,
                                                     loss_list)

    # Weight regularizer
    reg_loss = regularization_loss()

    # Total Loss
    total_loss = sqr_loss_f + reg_loss

    # Wasserstein Loss
    wass_loss = tf.constant(0.0)

    # Only add IPM if needed
    if opt_params.alpha > 0 or ("WASS" in loss_list and normalized_rep is not None):
        for t_id in range(1, n_treatments):
            wass_loss_t = wasserstein_loss(normalized_rep, t, p_t[t_id], t_id, lam=opt_params.wass_lambda,
                                           its=opt_params.wass_iterations,
                                           sq=False)

            wass_loss += wass_loss_t

        total_loss += opt_params.alpha * wass_loss

    losses_dict = {
        "MSE_F": sqr_loss_f,
        "MSE_CF": sqr_loss_cf,
        "WASS": wass_loss,
        "REG": reg_loss,
    }
    losses_dict.update(pre_m_losses_dict)

    return losses_dict
