import tensorflow as tf
import sonnet as snt


class _NN(snt.AbstractModule):
    def __init__(self, params, out_dim, name="NN"):
        super(_NN, self).__init__(name=name)
        self._out_dim = out_dim
        self._dim = params.dim
        self._n = params.n
        self._dropout = params.dropout
        self._batch_norm = params.batch_norm

        self._nonlinearity = tf.nn.relu

        self._regularizers = {"w": tf.contrib.layers.l2_regularizer(scale=params.weight_decay),
                              "b": tf.contrib.layers.l2_regularizer(scale=params.weight_decay)}
        self._initializers = {"w": tf.contrib.layers.xavier_initializer(),
                              "b": tf.constant_initializer(0)}

    def _build(self, x, is_training):
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

        current_in = x

        bn = [snt.BatchNorm() for i in range(self._n)]
        lin = [snt.Linear(output_size=self._dim, regularizers=self._regularizers, initializers=self._initializers) for i
               in range(self._n)]
        projection = snt.Linear(output_size=self._out_dim, regularizers=self._regularizers,
                                initializers=self._initializers)

        for i in range(self._n):
            current_in = _build_layer(current_in, self._batch_norm, self._dropout, bn[i], lin[i])
        out_projection = projection(current_in)

        return out_projection


class MultipleNN(snt.AbstractModule):
    '''Collection of independent NN for each output.'''
    def __init__(self, params, n_outputs, name="multipleNN"):
        super(MultipleNN, self).__init__(name=name)
        self._n_outputs = n_outputs
        self._params = params

    def _build(self, x, is_training):
        all_mu_hat = []
        all_sig_hat = []
        for t in range(self._n_outputs):
            model_mu = _NN(self._params, 1, "NN_mu")
            model_sig = _NN(self._params, 1, "NN_sig")
            mu_hat = model_mu(x, is_training)
            sig_hat = model_sig(x, is_training)

            all_mu_hat.append(mu_hat)
            all_sig_hat.append(sig_hat)

        mu_hat = tf.concat(all_mu_hat, axis=1)
        sig_hat = tf.concat(all_sig_hat, axis=1)

        return mu_hat, tf.pow(sig_hat, 2)
