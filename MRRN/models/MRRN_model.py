import tensorflow as tf
import sonnet as snt

import utils


class MRRN(snt.AbstractModule):
    def __init__(self, params, n_treatments, name="MRRN"):
        super().__init__(name=name)
        self._dim_in = params.dim_in
        self._dim_out = params.dim_out
        self._dim_rep = params.dim_rep if params.different_dim_rep else params.dim_in
        self._n_in = params.n_in
        self._n_out = params.n_out
        self._n_treatments = n_treatments
        self._dropout = params.dropout
        self._batch_norm_in = params.batch_norm_in
        self._batch_norm_out = params.batch_norm_out
        self._repeat_concat = params.repeat_concat

        if params.nonlinearity is "elu":
            self._nonlinearity = tf.nn.elu
        else:
            self._nonlinearity = tf.nn.relu

        self._regularizers = {"w": tf.contrib.layers.l2_regularizer(scale=params.weight_decay),
                              "b": tf.contrib.layers.l2_regularizer(scale=params.weight_decay)}
        self._initializers_in = {"w": tf.contrib.layers.xavier_initializer(),
                              "b": tf.constant_initializer(0)}
        self._initializers_out = {"w": tf.contrib.layers.xavier_initializer(),
                              "b": tf.constant_initializer(0)}

    def _build(self, x, t, s, is_training, treatment_types_all=None, skip_bin=False):
        def _build_layer(x, n, batch_norm, dropout, initializers_lin):
            if batch_norm:
                bn = snt.BatchNorm()(x, is_training=is_training)
            else:
                bn = x
            lin = snt.Linear(output_size=n, regularizers=self._regularizers, initializers=initializers_lin)(bn)
            out = self._nonlinearity(lin)
            if dropout < 1:
                out = tf.nn.dropout(out, dropout)
            return out

        # IN LAYERS
        current_in = x[:, :, 0]
        for i in range(self._n_in):
            n_nodes = self._dim_rep if i == (self._n_in-1) else self._dim_in
            current_in = _build_layer(current_in, n_nodes, self._batch_norm_in, self._dropout, self._initializers_in)

        # Representation
        rep = current_in  # Shape: bs x len(x)
        normalized_rep = rep / utils.safe_sqrt(tf.reduce_sum(tf.square(rep), axis=1, keepdims=True))
        current_in = normalized_rep

        # OUT LAYERS
        outputs = []
        current_param_treatment = 0
        for treatment in range(self._n_treatments):
            id_col_s = current_param_treatment if skip_bin else treatment

            s_to_append = s[:, id_col_s]  # bs x 1
            treatment_in = current_in

            if treatment_types_all[treatment] == 1:
                current_param_treatment += 1
            for i in range(self._n_out):
                if i == 0 or self._repeat_concat:
                    # append treatment strength: stack along dimension x
                    treatment_in = tf.concat([treatment_in, s_to_append], axis=1)
                treatment_in = _build_layer(treatment_in, self._dim_out, self._batch_norm_out, self._dropout, self._initializers_out)
            out_projection = snt.Linear(1, regularizers=self._regularizers, initializers=self._initializers_out)(treatment_in)
            outputs.append(out_projection)

        outputs = tf.concat(outputs, axis=1)

        # Only return parametric outputs
        if skip_bin:
            p_cols = []
            for i in range(self._n_treatments):
                if treatment_types_all[i]:
                    p_cols.append(i)
            outputs = tf.gather(outputs, p_cols, axis=1)

        return outputs, normalized_rep  # shapes: bs x n_treatments, bs x dim_in
