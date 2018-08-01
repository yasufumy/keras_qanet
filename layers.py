import math

import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Conv1D, Lambda, Dropout, SeparableConv1D, BatchNormalization


class SequenceLength(Lambda):
    def __init__(self, **kwargs):
        def func(x):
            mask = tf.cast(x, tf.bool)
            length = tf.expand_dims(tf.reduce_sum(tf.to_int32(mask), axis=1), axis=1)
            return length

        super().__init__(function=func, **kwargs)

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        return tf.TensorShape([batch_size, 1])


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

    def call(self, x):
        return self.add_timing_signal_1d(x)

    def compute_output_shape(self, input_shape):
        return input_shape


class MultiHeadAttention(Layer):
    def __init__(self, input_dim, num_heads, initializer, regularizer, dropout, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.initializer = initializer
        self.regularizer = regularizer
        self.d = input_dim // num_heads

    def build(self, input_shape):
        # W_Q: (filter_dim, input_dim, input_dim)
        self.W_Q = self.add_weight(
            'W_Q', [1, self.input_dim, self.input_dim], trainable=True,
            initializer=self.initializer, regularizer=self.regularizer)
        self.W_K = self.add_weight(
            'W_K', [1, self.input_dim, self.input_dim], trainable=True,
            initializer=self.initializer, regularizer=self.regularizer)
        self.W_V = self.add_weight(
            'W_V', [1, self.input_dim, self.input_dim], trainable=True,
            initializer=self.initializer, regularizer=self.regularizer)
        self.W_O = self.add_weight(
            'W_O', [1, self.input_dim, self.input_dim], trainable=True,
            initializer=self.initializer, regularizer=self.regularizer)

    def call(self, inputs, training=None):
        q, k, v, seq_len = inputs

        q = self.split_heads(K.conv1d(q, self.W_Q), self.num_heads)
        k = self.split_heads(K.conv1d(k, self.W_K), self.num_heads)
        v = self.split_heads(K.conv1d(v, self.W_V), self.num_heads)

        scale = self.d ** (1/2)
        q *= scale
        x = self.dot_product_attention(q, k, v, seq_len, self.dropout, training)
        x = self.combine_heads(x)
        return K.conv1d(x, self.W_O)

    def split_heads(self, x, n):
        shape = x.shape.as_list()
        splitted = tf.reshape(x, [-1] + shape[1:-1] + [n, shape[-1] // n])
        return tf.transpose(splitted, [0, 2, 1, 3])

    def combine_heads(self, x):
        x = tf.transpose(x, [0, 2, 1, 3])
        shape = x.shape.as_list()
        return tf.reshape(x, [-1] + shape[1:-2] + [shape[-2] * shape[-1]])

    def dot_product_attention(self, q, k, v, seq_len, dropout=.1, training=None):
        logits = tf.matmul(q, k, transpose_b=True)
        # logits = self.mask_logits(logits, seq_len)
        # weights = tf.nn.softmax(logits, axis=-1)
        weights = self.masked_softmax(logits, seq_len, axis=-1)
        weights = K.in_train_phase(tf.nn.dropout(weights, 1 - dropout), weights, training=training)
        return tf.matmul(weights, v)

    def masked_softmax(self, x, mask, axis=-1, mask_value=tf.float32.min):
        maxlen = x.shape.as_list()[-1]
        # mask: (batch, 1, seq_len)
        mask = tf.sequence_mask(mask, maxlen=maxlen, dtype=tf.float32)
        mask = tf.expand_dims(tf.matmul(mask, mask, transpose_a=True), axis=1)  # (batch, 1, seq_len, seq_len)
        x = x + (1 - mask) * mask_value
        weights = tf.nn.softmax(x, axis=axis)
        return weights * mask

    def compute_output_shape(self, input_shape):
        return input_shape[1]


class ContextQueryAttention(Layer):
    def __init__(self, cont_limit, ques_limit, initializer, regularizer, dropout, **kwargs):
        super().__init__(**kwargs)
        self.cont_limit = cont_limit
        self.ques_limit = ques_limit
        self.initializer = initializer
        self.regularizer = regularizer
        self.dropout = dropout

    def build(self, input_shape):
        # (batch, seq_len, hidden_dim)
        c_shape = input_shape[0]
        self.W = self.add_weight(
            'weight', [1, c_shape[-1] * 3, 1], trainable=True,
            initializer=self.initializer, regularizer=self.regularizer)

    def call(self, inputs):
        c, q, c_len, q_len = inputs
        d = c.shape[-1]  # hidden_dim

        # similarity
        c_tile = tf.tile(tf.expand_dims(c, 2), [1, 1, self.ques_limit, 1])
        q_tile = tf.tile(tf.expand_dims(q, 1), [1, self.cont_limit, 1, 1])
        total_len = self.ques_limit * self.cont_limit
        c_mat = tf.reshape(c_tile, [-1, total_len, d])
        q_mat = tf.reshape(q_tile, [-1, total_len, d])
        c_q = c_mat * q_mat
        weight_in = tf.concat([c_mat, q_mat, c_q], 2)
        S = tf.reshape(K.conv1d(weight_in, self.W), [-1, self.cont_limit, self.ques_limit])

        # mask
        # mask: (batch, 1, c_len)
        c_mask = tf.sequence_mask(c_len, maxlen=self.cont_limit, dtype=tf.float32)
        # mask: (batch, 1, q_len)
        q_mask = tf.sequence_mask(q_len, maxlen=self.ques_limit, dtype=tf.float32)
        # mask: (batch, c_len, q_len)
        mask = tf.matmul(c_mask, q_mask, transpose_a=True)

        # softmax
        S_q = self.masked_softmax(S, mask, axis=2)
        S_c = self.masked_softmax(S, mask, axis=1)
        a = tf.matmul(S_q, q)
        b = tf.matmul(tf.matmul(S_q, S_c, transpose_b=True), c)
        x = tf.concat([c, a, c * a, c * b], axis=2)
        return [x, S_q, S_c]

    def masked_softmax(self, x, mask, axis=-1, mask_value=tf.float32.min):
        x = x + (1 - mask) * mask_value
        weights = tf.nn.softmax(x, axis=axis)
        return weights * mask

    def compute_output_shape(self, input_shape):
        batch, c_len, d = input_shape[0]
        batch, q_len, d = input_shape[1]
        return [(batch, c_len, d * 4), (batch, c_len, q_len), (batch, c_len, q_len)]


class LayerDropout(Layer):
    def __init__(self, dropout=0., **kwargs):
        self.dropout = dropout
        super().__init__(**kwargs)

    def call(self, inputs, training=None):
        x, residual = inputs
        pred = tf.random_uniform([]) < self.dropout
        x_train = tf.cond(
            pred, lambda: residual, lambda: tf.nn.dropout(x, 1. - self.dropout) + residual)
        x_test = x + residual
        return K.in_train_phase(x_train, x_test, training=training)

    def compute_output_shape(self, input_shape):
        return input_shape


class Highway:
    def __init__(self, filters, num_layers, initializer=None, regularizer=None, dropout=.1):
        self.dropout = dropout
        conv_layers = []
        for i in range(num_layers):
            conv_layers.append([
                Conv1D(filters, 1, activation='sigmoid', kernel_initializer=initializer,
                       kernel_regularizer=regularizer, bias_regularizer=regularizer),
                Conv1D(filters, 1, activation='relu', kernel_initializer=initializer,
                       kernel_regularizer=regularizer, bias_regularizer=regularizer)])
        self.conv_layers = conv_layers

    def __call__(self, x):
        conv_layers = self.conv_layers
        for i in range(len(conv_layers)):
            T = conv_layers[i][0](x)
            H = conv_layers[i][1](x)
            H = Dropout(self.dropout)(x)
            x = Lambda(lambda inputs: inputs[0] * inputs[1] + inputs[2] * (1 - inputs[1]))([H, T, x])
        return x


class Encoder:
    def __init__(self, filters, kernel_size, num_blocks, num_convs, num_heads,
                 initializer=None, regularizer=None, dropout=.1):
        conv_layers = []
        attention_layers = []
        feedforward_layers = []
        for i in range(num_blocks):
            conv_layers.append([])
            for j in range(num_convs):
                conv_layers[i].append(
                    SeparableConv1D(
                        filters, 7, padding='same', depthwise_initializer=initializer,
                        pointwise_initializer=initializer, depthwise_regularizer=regularizer,
                        pointwise_regularizer=regularizer, activation='relu',
                        bias_regularizer=regularizer, activity_regularizer=regularizer))
            attention_layers.append(
                MultiHeadAttention(filters, num_heads, initializer, regularizer, dropout))
            feedforward_layers.append([
                Conv1D(filters, 1, activation='relu', kernel_initializer=initializer,
                       kernel_regularizer=regularizer, bias_regularizer=regularizer),
                Conv1D(filters, 1, activation='linear', kernel_initializer=initializer,
                       kernel_regularizer=regularizer, bias_regularizer=regularizer)])

        self.conv_layers = conv_layers
        self.attention_layers = attention_layers
        self.feedforward_layers = feedforward_layers
        self.num_blocks = num_blocks
        self.num_convs = num_convs
        self.dropout = dropout

    def __call__(self, x, seq_len):
        conv_layers = self.conv_layers
        attention_layers = self.attention_layers
        feedforward_layers = self.feedforward_layers
        num_blocks = self.num_blocks
        num_convs = self.num_convs
        dropout = self.dropout
        total_layer = (2 + num_convs) * num_blocks
        sub_layer = 1

        for i in range(num_blocks):
            x = PositionEmbedding()(x)
            # convolution
            for j in range(num_convs):
                residual = x
                x = BatchNormalization()(x)
                if sub_layer % 2 == 0:
                    x = Dropout(dropout)(x)
                x = conv_layers[i][j](x)
                x = LayerDropout(dropout * (sub_layer / total_layer))([x, residual])
                sub_layer += 1
            # attention
            residual = x
            x = BatchNormalization()(x)
            if sub_layer % 2 == 0:
                x = Dropout(dropout)(x)
            x = attention_layers[i]([x, x, x, seq_len])
            x = LayerDropout(dropout * (sub_layer / total_layer))([x, residual])
            # feed-forward
            residual = x
            x = BatchNormalization()(x)
            if sub_layer % 2 == 0:
                x = Dropout(dropout)(x)
            x = feedforward_layers[i][0](x)
            x = feedforward_layers[i][1](x)
            x = LayerDropout(dropout * (sub_layer / total_layer))([x, residual])
            sub_layer += 1
        return x
