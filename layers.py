import math

import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Conv1D, Lambda, Dropout
from keras.regularizers import l2


class PositionEmbedding(Layer):
    def __init__(self, min_timescale=1., max_timescale=1.e4, **kwargs):
        self.min_timescale = float(min_timescale)
        self.max_timescale = float(max_timescale)
        super().__init__(**kwargs)

    def get_timing_signal_1d(self, length, channels):
        position = tf.to_float(tf.range(length))
        num_timescales = channels // 2
        log_timescale_increment = \
            math.log(self.max_timescale / self.min_timescale) / \
            (tf.to_float(num_timescales) - 1)
        inv_timescales = self.min_timescale * \
            tf.exp(tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
        signal = tf.reshape(signal, [1, length, channels])
        return signal

    def add_timing_signal_1d(self, x):
        length = tf.shape(x)[1]  # sequence length
        channels = tf.shape(x)[2]  # hidden dimension for each word
        signal = self.get_timing_signal_1d(length, channels)
        return x + signal

    def call(self, x, mask=None):
        return self.add_timing_signal_1d(x)

    def compute_output_shape(self, input_shape):
        return input_shape


class MultiHeadAttention(Layer):
    def __init__(self, hidden_size, num_heads, dropout=0.0, bias=False, **kwargs):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.bias = bias
        super().__init__(**kwargs)

    def build(self, input_shape):
        if self.bias:
            self.b = self.add_weight(
                name='bias', shape=(input_shape[0][-2],), initializer='zero')

        super().build(input_shape)

    def split_last_dim(self, x, n):
        old_shape = x.get_shape().dims  # batch_size * seq_len * hidden_size
        last = old_shape[-1]  # last shape should be hidden dimension
        # batch_size * seq_len * heads * hidden_size // heads
        new_shape = old_shape[:-1] + [n] + [last // n if last else None]
        # reshape batch_size * seq_len * heads * hidden_size // heads
        ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
        ret.set_shape(new_shape)
        # reshape batch_size * heads * seq_len * hidden_size // heads
        return tf.transpose(ret, [0, 2, 1, 3])

    def mask_logits(self, inputs, mask, mask_value=tf.float32.min):
        shapes = [x if x is not None else -1 for x in inputs.shape.as_list()]
        mask = K.cast(mask, tf.int32)
        mask = K.one_hot(mask[:, 0], shapes[-1])
        mask = 1 - K.cumsum(mask, 1)
        mask = tf.cast(mask, tf.float32)
        mask = tf.reshape(mask, [shapes[0], 1, 1, shapes[-1]])
        return inputs + mask_value * (1 - mask)

    def dot_product_attention(self, x, seq_len=None, dropout=.1, training=None):
        q, k, v = x
        logits = tf.matmul(q, k, transpose_b=True)
        if self.bias:
            logits += self.b
        if seq_len is not None:
            logits = self.mask_logits(logits, seq_len)
        weights = tf.nn.softmax(logits)
        weights = K.in_train_phase(K.dropout(weights, dropout), weights, training=training)
        return tf.matmul(weights, v)

    def combine_last_two_dims(self, x):
        old_shape = x.get_shape().dims
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        ret = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
        ret.set_shape(new_shape)
        return ret

    def call(self, x, mask=None, training=None):
        key_and_value, query, seq_len = x
        Q = self.split_last_dim(query, self.num_heads)
        key, value = tf.split(key_and_value, 2, axis=2)
        K = self.split_last_dim(key, self.num_heads)
        V = self.split_last_dim(value, self.num_heads)

        scale = (self.hidden_size // self.num_heads) ** (1/2)
        Q *= scale
        x = self.dot_product_attention([Q, K, V], seq_len, dropout=self.dropout, training=training)
        x = self.combine_last_two_dims(tf.transpose(x, [0, 2, 1, 3]))
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[1]


class ContextQueryAttention(Layer):
    def __init__(self, output_size, cont_limit, ques_limit, dropout=0.0, **kwargs):
        self.output_size = output_size
        self.cont_limit = cont_limit
        self.ques_limit = ques_limit
        self.dropout = dropout
        super().__init__(**kwargs)

    def build(self, input_shape):
        # input_shape = [(batch, context_limit, hidden), (batch, question_limit, hidden)]
        self.W0 = self.add_weight(name='W0', trainable=True, shape=(input_shape[0][2], 1),
                                  initializer='glorot_uniform', regularizer=l2(3e-7))
        self.W1 = self.add_weight(name='W1', trainable=True, shape=(input_shape[1][2], 1),
                                  initializer='glorot_uniform', regularizer=l2(3e-7))
        self.W2 = self.add_weight(name='W2', trainable=True, shape=(1, 1, input_shape[0][2]),
                                  initializer='glorot_uniform', regularizer=l2(3e-7))

    def apply_mask(self, inputs, seq_len, axis=1, time_dim=1, mode='add'):
        if seq_len is None:
            return inputs
        else:
            seq_len = K.cast(seq_len, tf.int32)
            mask = K.one_hot(seq_len[:, 0], K.shape(inputs)[time_dim])
            mask = 1 - K.cumsum(mask, 1)
            mask = K.expand_dims(mask, axis)
            if mode == 'add':
                return inputs - (1 - mask) * 1e12
            if mode == 'mul':
                return inputs * mask

    def call(self, inputs, mask=None):
        x_cont, x_ques, cont_len, ques_len = inputs
        S_cont = K.tile(K.dot(x_cont, self.W0), [1, 1, self.ques_limit])
        S_ques = K.tile(K.permute_dimensions(K.dot(x_ques, self.W1), pattern=(0, 2, 1)), [1, self.cont_limit, 1])
        S_fuse = K.batch_dot(x_cont * self.W2, K.permute_dimensions(x_ques, pattern=(0, 2, 1)))
        S = S_cont + S_ques + S_fuse
        S_bar = tf.nn.softmax(self.apply_mask(S, ques_len, axis=1, time_dim=2))
        S_T = K.permute_dimensions(tf.nn.softmax(self.apply_mask(S, cont_len, axis=2, time_dim=1), axis=1), (0, 2, 1))
        c2q = tf.matmul(S_bar, x_ques)
        q2c = tf.matmul(tf.matmul(S_bar, S_T), x_cont)
        result = K.concatenate([x_cont, c2q, x_cont * c2q, x_cont * q2c], axis=-1)
        return [result, S_bar, S_T]

    def compute_output_shape(self, input_shape):
        return [(input_shape[0][0], input_shape[0][1], self.output_size),
                (input_shape[0][0], input_shape[0][1], input_shape[1][1]),
                (input_shape[0][0], input_shape[1][1], input_shape[0][1])]


class LayerDropout(Layer):
    def __init__(self, dropout=0., **kwargs):
        self.dropout = dropout
        super().__init__(**kwargs)

    def call(self, inputs, mask=None, training=None):
        x, residual = inputs
        pred = tf.random_uniform([]) < self.dropout
        x_train = tf.cond(
            pred, lambda: residual, lambda: tf.nn.dropout(x, 1. - self.dropout) + residual)
        x_test = x + residual
        return K.in_train_phase(x_train, x_test, training=training)

    def compute_output_shape(self, input_shape):
        return input_shape


class Highway(Layer):
    def __init__(self, output_dim, num_layers, regularizer=None, dropout=0., **kwargs):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.output_dim = output_dim
        self.squeeze_layer = Conv1D(output_dim, 1, kernel_regularizer=regularizer, activation='linear')
        conv_layers = []
        for i in range(num_layers):
            conv_layers.append([Conv1D(output_dim, 1, kernel_regularizer=regularizer, activation='sigmoid'),
                                Conv1D(output_dim, 1, kernel_regularizer=regularizer, activation='linear')])
        self.conv_layers = conv_layers

    def call(self, x):
        x = self.squeeze_layer(x)
        conv_layers = self.conv_layers
        for i in range(len(conv_layers)):
            T = conv_layers[i][0](x)
            H = conv_layers[i][1](x)
            H = Dropout(self.dropout)(x)
            x = Lambda(lambda inputs: inputs[0] * inputs[1] + inputs[2] * (1 - inputs[1]))([H, T, x])
        return x

    def compute_output_shape(self, input_shape):
        output_shape = input_shape[:-1] + (self.output_dim,)
        return output_shape
