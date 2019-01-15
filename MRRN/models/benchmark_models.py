import copy

import numpy as np
import tensorflow as tf
import sonnet as snt

import data_loader


class OLS(snt.AbstractModule):
    def __init__(self, params, n_treatments, name="OLS"):
        super(OLS, self).__init__(name=name)
        self._n_treatments = n_treatments
        self.concat_treatment = params.concat_treatment
        self.regularizers = {"w": tf.contrib.layers.l1_l2_regularizer(
            scale_l1=float(params.weight_decay_l1), scale_l2=float(params.weight_decay_l2))}

    def _build(self, x, t, s, is_training, treatment_types_all=None, skip_bin=False):
        current_in = x[:, :, 0]

        outputs = []
        current_param_treatment = 0

        projection_layer = snt.Linear(1, regularizers=self.regularizers)

        for treatment in range(self._n_treatments):
            id_col_s = current_param_treatment if skip_bin else treatment

            treatment_in = tf.concat([current_in, s[:, id_col_s]], axis=1)  # append treatment strength along dimension x
            if self.concat_treatment:
                treatment_in = tf.concat([treatment_in, tf.to_float(t)], axis=1)  # append treatment t

            if treatment_types_all[treatment] == 1:
                current_param_treatment += 1

            out_projection = projection_layer(treatment_in)
            outputs.append(out_projection)

        outputs = tf.concat(outputs, axis=1)

        if skip_bin:
            p_cols = []
            for i in range(self._n_treatments):
                if treatment_types_all[i]:
                    p_cols.append(i)
            outputs = tf.gather(outputs, p_cols, axis=1)

        return outputs, None  # shapes: bs x n_treatments, none


class NN(snt.AbstractModule):
    def __init__(self, params, n_treatments, name="NN"):
        super(NN, self).__init__(name=name)
        self._n_treatments = n_treatments
        self._dim = params.dim
        self._n = params.n
        self._dropout = params.dropout
        self._batch_norm = params.batch_norm

        if params.nonlinearity is "elu":
            self._nonlinearity = tf.nn.elu
        else:
            self._nonlinearity = tf.nn.relu

        self._regularizers = {"w": tf.contrib.layers.l2_regularizer(scale=params.weight_decay),
                              "b": tf.contrib.layers.l2_regularizer(scale=params.weight_decay)}
        std = 0.1 / np.sqrt(params.dim)
        self._initializers = {"w": tf.truncated_normal_initializer(stddev=std),
                              "b": tf.constant_initializer(0)}

    def _build(self, x, t, s, is_training, treatment_types_all=None, skip_bin=False):
        def _build_layer(x, batch_norm, dropout, premade_batchnorm, premade_linear):
            if batch_norm:
                bn = premade_batchnorm(x, is_training=is_training)
            else:
                bn = x
            lin = premade_linear(bn)
            out = self._nonlinearity(lin)
            if dropout < 1:
                out = tf.nn.dropout(out, dropout)
            return out

        current_in = x[:, :, 0]

        outputs = []
        current_param_treatment = 0

        bn = [snt.BatchNorm() for _ in range(self._n)]
        lin = [snt.Linear(output_size=self._dim, regularizers=self._regularizers, initializers=self._initializers) for i in range(self._n)]
        projection = snt.Linear(output_size=1, regularizers=self._regularizers, initializers=self._initializers)

        for treatment in range(self._n_treatments): #  TODO: version without head-splitting, but rather a n_treatment dimensional output
            id_col_s = current_param_treatment if skip_bin else treatment

            treatment_in = tf.concat([current_in, s[:, id_col_s]], axis=1)  # append treatment strength along dimension x
            treatment_in = tf.concat([treatment_in, tf.to_float(t)], axis=1)  # append treatment t

            if treatment_types_all[treatment] == 1:
                current_param_treatment += 1

            for i in range(self._n):
                treatment_in = _build_layer(treatment_in, self._batch_norm, self._dropout, bn[i], lin[i])
            out_projection = projection(treatment_in)
            outputs.append(out_projection)
        outputs = tf.concat(outputs, axis=1)

        if skip_bin:
            p_cols = []
            for i in range(self._n_treatments):
                if treatment_types_all[i]:
                    p_cols.append(i)
            outputs = tf.gather(outputs, p_cols, axis=1)

        return outputs, None  # shapes: bs x n_treatments, none


class SplitRegressor(snt.AbstractModule):
    """
    Regressor wrapper that splits multi treatment problem into n_treatments regressions.
    """
    def __init__(self, params, n_treatments, name="SplitRegressor", regressor_class=OLS):
        super(SplitRegressor, self).__init__(name=name)
        self._n_treatments = n_treatments

        # Different treatments are handled by splitting
        p = copy.deepcopy(params)
        if p.concat_treatment:
            p.concat_tratment = False
        self._params = p

        self._regressor_class = regressor_class

    def _build(self, x, t, s, is_training, treatment_types_all, skip_bin=False):
        outputs_list = []
        p_treatment_counter = 0
        for t_id in range(self._n_treatments):
            id_col_s = p_treatment_counter if skip_bin else t_id

            if treatment_types_all[t_id] == 1:
                p_treatment_counter += 1

            o, _ = self._regressor_class(self._params, 1)(x, tf.zeros_like(t), s[:, id_col_s, tf.newaxis], is_training, [treatment_types_all[t_id]])

            outputs_list.append(o)

        outputs = tf.concat(outputs_list, axis=1)

        if skip_bin:
            p_cols = []
            for i in range(self._n_treatments):
                if treatment_types_all[i]:
                    p_cols.append(i)
            outputs = tf.gather(outputs, p_cols, axis=1)

        return outputs, None
