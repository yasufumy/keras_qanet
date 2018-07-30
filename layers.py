import math

import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Conv1D, Lambda, Dropout, SeparableConv1D, BatchNormalization
from keras.initializers import VarianceScaling


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
    def __init__(self, input_dim, num_heads, dropout, regularizer, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.regularizer = regularizer
        self.d = input_dim // num_heads

    def build(self, input_shape):
        self.W_Q = self.add_weight(
            'W_Q', [self.input_dim, self.input_dim], trainable=True,
            initializer=VarianceScaling(scale=1., mode='fan_in', distribution='normal'),
            regularizer=self.regularizer)
        self.W_K = self.add_weight(
            'W_K', [self.input_dim, self.input_dim], trainable=True,
            initializer=VarianceScaling(scale=1., mode='fan_in', distribution='normal'),
            regularizer=self.regularizer)
        self.W_V = self.add_weight(
            'W_V', [self.input_dim, self.input_dim], trainable=True,
            initializer=VarianceScaling(scale=1., mode='fan_in', distribution='normal'),
            regularizer=self.regularizer)
        self.W_O = self.add_weight(
            'W_O', [self.input_dim, self.input_dim], trainable=True,
            initializer=VarianceScaling(scale=1., mode='fan_in', distribution='normal'),
            regularizer=self.regularizer)

    def call(self, inputs, training=None):
        q, k, v, seq_len = inputs

        q = self.split_heads(K.dot(q, self.W_Q), self.num_heads)
        k = self.split_heads(K.dot(k, self.W_K), self.num_heads)
        v = self.split_heads(K.dot(v, self.W_V), self.num_heads)

        scale = self.d ** (1/2)
        q *= scale
        x = self.dot_product_attention(q, k, v, seq_len, self.dropout, training)
        x = self.combine_heads(x)
        return K.dot(x, self.W_O)

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
        logits = self.mask_logits(logits, seq_len)
        weights = tf.nn.softmax(logits, axis=-1)
        weights = K.in_train_phase(tf.nn.dropout(weights, 1 - dropout), weights, training=training)
        return tf.matmul(weights, v)

    def mask_logits(self, x, mask, mask_value=tf.float32.min):
        # shapes = [x if x is not None else -1 for x in inputs.shape.as_list()]
        # mask = K.cast(mask, tf.int32)
        # mask = K.one_hot(mask[:, 0], shapes[-1])
        # mask = 1 - K.cumsum(mask, 1)
        # mask = tf.cast(mask, tf.float32)
        # mask = tf.reshape(mask, [shapes[0], 1, 1, shapes[-1]])
        maxlen = x.shape.as_list()[-1]
        # mask: (batch, 1, seq_len)
        mask = tf.sequence_mask(mask, maxlen=maxlen, dtype=tf.float32)
        mask = tf.expand_dims(tf.matmul(mask, mask, transpose_a=True), axis=1)  # (batch, 1, seq_len, seq_len)
        mask = tf.where(tf.is_nan(mask), x=tf.zeros_like(mask), y=mask)
        # mask = tf.expand_dims(mask, axis=1)
        return x + mask_value * (1 - mask)

    def compute_output_shape(self, input_shape):
        return input_shape[1]


class ContextQueryAttention(Layer):
    def __init__(self, output_size, cont_limit, ques_limit, dropout, regularizer, **kwargs):
        self.output_size = output_size
        self.cont_limit = cont_limit
        self.ques_limit = ques_limit
        self.dropout = dropout
        self.regularizer = regularizer
        super().__init__(**kwargs)

    def build(self, input_shape):
        # input_shape = [(batch, context_limit, hidden), (batch, question_limit, hidden)]
        self.W0 = self.add_weight(name='W0', trainable=True, shape=(input_shape[0][2], 1),
                                  initializer=VarianceScaling(scale=1., mode='fan_in', distribution='normal'),
                                  regularizer=self.regularizer)
        self.W1 = self.add_weight(name='W1', trainable=True, shape=(input_shape[1][2], 1),
                                  initializer=VarianceScaling(scale=1., mode='fan_in', distribution='normal'),
                                  regularizer=self.regularizer)
        self.W2 = self.add_weight(name='W2', trainable=True, shape=(1, 1, input_shape[0][2]),
                                  initializer=VarianceScaling(scale=1., mode='fan_in', distribution='normal'),
                                  regularizer=self.regularizer)

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

    def call(self, inputs):
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

    def call(self, inputs, training=None):
        x, residual = inputs
        pred = tf.random_uniform([]) < self.dropout
        x_train = tf.cond(
            pred, lambda: residual, lambda: tf.nn.dropout(x, 1. - self.dropout) + residual)
        x_test = x + residual
        return K.in_train_phase(x_train, x_test, training=training)

    def compute_output_shape(self, input_shape):
        return input_shape


class Highway(Layer):
    def __init__(self, filters, num_layers, regularizer=None, dropout=0., **kwargs):
        super().__init__(**kwargs)

        self.dropout = dropout
        conv_layers = []
        for i in range(num_layers):
            conv_layers.append([Conv1D(filters, 1, kernel_regularizer=regularizer, activation='sigmoid'),
                                Conv1D(filters, 1, kernel_regularizer=regularizer, activation='relu')])
        self.conv_layers = conv_layers

    def call(self, x, training=None):
        conv_layers = self.conv_layers
        for i in range(len(conv_layers)):
            T = conv_layers[i][0](x)
            H = conv_layers[i][1](x)
            H = Dropout(self.dropout)(x, training=training)
            x = Lambda(lambda inputs: inputs[0] * inputs[1] + inputs[2] * (1 - inputs[1]))([H, T, x])
        return x


class Encoder(Layer):
    def __init__(self, filters, kernel_size, num_blocks, num_convs, num_heads, dropout, regularizer, **kwargs):
        super().__init__(**kwargs)

        conv_layers = []
        attention_layers = []
        feedforward_layers = []
        for i in range(num_blocks):
            conv_layers.append([])
            for j in range(num_convs):
                conv_layers[i].append(SeparableConv1D(
                    filters, kernel_size, padding='same', activation='relu',
                    depthwise_regularizer=regularizer, pointwise_regularizer=regularizer,
                    bias_regularizer=regularizer, activity_regularizer=regularizer))
            attention_layers.append(
                MultiHeadAttention(filters, num_heads, dropout, regularizer))
            feedforward_layers.append([
                Conv1D(filters, 1, activation='relu', kernel_regularizer=regularizer),
                Conv1D(filters, 1, activation='linear', kernel_regularizer=regularizer)])

        self.dropout = dropout
        self.num_blocks = num_blocks
        self.num_convs = num_convs
        self.conv_layers = conv_layers
        self.attention_layers = attention_layers
        self.feedforward_layers = feedforward_layers

    def call(self, inputs, training=None):
        x, seq_len = inputs
        conv_layers = self.conv_layers
        attention_layers = self.attention_layers
        feedforward_layers = self.feedforward_layers
        dropout = self.dropout
        num_blocks = self.num_blocks
        num_convs = self.num_convs
        total_layer = (2 + num_convs) * num_blocks
        sub_layer = 1

        for i in range(num_blocks):
            x = PositionEmbedding()(x)
            # conv
            for j in range(num_convs):
                residual = x
                x = BatchNormalization()(x, training=training)
                if sub_layer % 2 == 0:
                    x = Dropout(dropout)(x, training=training)
                x = conv_layers[i][j](x)
                x = LayerDropout(dropout * (sub_layer / total_layer))([x, residual], training=training)
                sub_layer += 1
            # attention
            residual = x
            x = BatchNormalization()(x, training=training)
            if sub_layer % 2 == 0:
                x = Dropout(dropout)(x, training=training)
            x = attention_layers[i]([x, x, x, seq_len], training=training)
            x = LayerDropout(dropout * (sub_layer / total_layer))([x, residual], training=training)
            sub_layer += 1
            # feed-forward
            residual = x
            x = BatchNormalization()(x, training=training)
            if sub_layer % 2 == 0:
                x = Dropout(dropout)(x, training=training)
            x = feedforward_layers[i][0](x)
            x = feedforward_layers[i][1](x)
            x = LayerDropout(dropout * (sub_layer / total_layer))([x, residual], training=training)
            sub_layer += 1
        return x
