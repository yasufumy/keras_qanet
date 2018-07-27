import tensorflow as tf
from keras import backend as K
from keras import Model
from keras.regularizers import l2
from keras.initializers import VarianceScaling
from keras.layers import Input, Embedding, Concatenate, Lambda, \
    BatchNormalization, Conv1D, Dropout, Masking, \
    LSTM, Bidirectional, Dense, SeparableConv2D

from layers import MultiHeadAttention, PositionEmbedding, ContextQueryAttention, \
    LayerDropout, Highway


regularizer = l2(3e-7)
VarianceScaling(scale=1., mode='fan_in', distribution='normal')


def squeeze_block(x, squeeze_layer, dropout=0.):
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    x = squeeze_layer(x)
    return x


def conv_block(x, conv_layers, dropout=0., sub_layer=1., last_layer=1.):
    n_conv = len(conv_layers)
    x = Lambda(lambda x: tf.expand_dims(x, axis=2))(x)
    for i in range(n_conv):
        residual = x
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        # x = conv_layers[i][0](x)
        # x = conv_layers[i][1](x)
        x = conv_layers[i](x)
        x = LayerDropout(dropout * (sub_layer / last_layer))([x, residual])
    x = Lambda(lambda x: tf.squeeze(x, axis=2))(x)
    return x


def attention_block(x, attention_layer, seq_len, dropout=0., sub_layer=1., last_layer=1.):
    residual = x
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    key_and_value = attention_layer[0](x)
    query = attention_layer[1](x)
    x = attention_layer[2]([key_and_value, query, seq_len])
    return LayerDropout(dropout * (sub_layer / last_layer))([x, residual])


def ffn_block(x, ffn_layer, dropout=0., sub_layer=1., last_layer=1.):
    residual = x
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    x = ffn_layer[0](x)
    x = ffn_layer[1](x)
    return LayerDropout(dropout * (sub_layer / last_layer))([x, residual])


def encoder_block(x, conv_layers, attention_layer, ffn_layer, seq_len, dropout,
                  num_blocks, repeat=1):
    outputs = [x]
    for _ in range(repeat):
        x = outputs[-1]
        for j in range(num_blocks):
            x = PositionEmbedding()(x)
            x = conv_block(x, conv_layers[j], dropout, j, num_blocks)
            x = attention_block(x, attention_layer[j], seq_len, dropout, j, num_blocks)
            x = ffn_block(x, ffn_layer[j], dropout, j, num_blocks)
        outputs.append(x)
    return outputs


def mask_logits(inputs, mask, mask_value=tf.float32.min, axis=1, time_dim=1):
    mask = K.cast(mask, tf.int32)
    mask = K.one_hot(mask[:, 0], K.shape(inputs)[time_dim])
    mask = 1 - K.cumsum(mask, 1)
    mask = tf.cast(mask, tf.float32)
    if axis != 0:
        mask = tf.expand_dims(mask, axis)
    return inputs + mask_value * (1 - mask)


class QANet:
    def __init__(self, vocab_size, embed_size, filters=128, num_heads=1,
                 cont_limit=400, ques_limit=50,
                 dropout=0.1, encoder_layer_size=1, encoder_conv_blocks=2,
                 output_layer_size=2, output_conv_blocks=2, embeddings=None):
        self.cont_limit = cont_limit
        self.ques_limit = ques_limit
        self.dropout = dropout
        self.encoder_layer_size = encoder_layer_size
        self.output_layer_size = output_layer_size
        if embeddings is not None:
            embeddings = [embeddings]
        self.embed_layer = Embedding(
            vocab_size, embed_size, weights=embeddings, trainable=False)
        self.e2h_squeeze_layer = Conv1D(filters, 1)
        conv_layers = []
        self_attention_layer = []
        ffn_layer = []
        for i in range(encoder_layer_size):
            conv_layers.append([])
            for j in range(encoder_conv_blocks):
                conv_layers[i].append(SeparableConv2D(
                    filters=filters, kernel_size=(7, 1), padding='same', activation='relu',
                    depthwise_regularizer=regularizer, pointwise_regularizer=regularizer))
                # conv_layers[i].append([
                #     DepthwiseConv2D((7, 1), padding='same', depth_multiplier=1, kernel_regularizer=regularizer),
                #     Conv2D(filters, 1, padding='same', kernel_regularizer=regularizer)])
            self_attention_layer.append([
                Conv1D(2 * filters, 1, kernel_regularizer=regularizer),  # weights for key and value
                Conv1D(filters, 1, kernel_regularizer=regularizer),  # weights for query
                MultiHeadAttention(filters, num_heads)])
            ffn_layer.append([Conv1D(filters, 1, activation='relu', kernel_regularizer=regularizer),
                              Conv1D(filters, 1, activation='linear', kernel_regularizer=regularizer)])
        self.conv_layers = conv_layers
        self.self_attention_layer = self_attention_layer
        self.ffn_layer = ffn_layer

        self.cqattention_layer = ContextQueryAttention(filters * 4, cont_limit, ques_limit, dropout)
        self.h2o_squeeze_layer = Conv1D(filters, 1, activation='linear', kernel_regularizer=regularizer)

        conv_layers = []
        self_attention_layer = []
        ffn_layer = []
        for i in range(output_layer_size):
            conv_layers.append([])
            for j in range(output_conv_blocks):
                # conv_layers[i].append([
                #     DepthwiseConv2D((5, 1), padding='same', depth_multiplier=1, kernel_regularizer=regularizer),
                #     Conv2D(filters, 1, padding='same', kernel_regularizer=regularizer)])
                conv_layers[i].append(SeparableConv2D(
                    filters=filters, kernel_size=(7, 1), padding='same', activation='relu',
                    depthwise_regularizer=regularizer, pointwise_regularizer=regularizer))
            self_attention_layer.append([
                Conv1D(2 * filters, 1, kernel_regularizer=regularizer),
                Conv1D(filters, 1, kernel_regularizer=regularizer),
                MultiHeadAttention(filters, num_heads)])
            ffn_layer.append([Conv1D(filters, 1, activation='relu', kernel_regularizer=regularizer),
                              Conv1D(filters, 1, activation='linear', kernel_regularizer=regularizer)])
        self.conv_layers2 = conv_layers
        self.self_attention_layer2 = self_attention_layer
        self.ffn_layer2 = ffn_layer

        self.start_layer = Conv1D(1, 1, activation='linear', kernel_regularizer=regularizer)
        self.end_layer = Conv1D(1, 1, activation='linear', kernel_regularizer=regularizer)

    def build(self):
        dropout = self.dropout

        cont_input = Input((self.cont_limit,))
        ques_input = Input((self.ques_limit,))

        # mask
        c_mask = Lambda(lambda x: tf.cast(x, tf.bool))(cont_input)
        q_mask = Lambda(lambda x: tf.cast(x, tf.bool))(ques_input)
        cont_len = Lambda(lambda x: tf.expand_dims(tf.reduce_sum(tf.cast(x, tf.int32), axis=1), axis=1))(c_mask)
        ques_len = Lambda(lambda x: tf.expand_dims(tf.reduce_sum(tf.cast(x, tf.int32), axis=1), axis=1))(q_mask)

        # encoding each
        x_cont = self.embed_layer(cont_input)
        x_cont = squeeze_block(x_cont, self.e2h_squeeze_layer, dropout)
        x_cont = encoder_block(x_cont, self.conv_layers, self.self_attention_layer,
                               self.ffn_layer, cont_len, dropout,
                               num_blocks=self.encoder_layer_size, repeat=1)[1]

        x_ques = self.embed_layer(ques_input)
        x_ques = squeeze_block(x_ques, self.e2h_squeeze_layer, dropout)
        x_ques = encoder_block(x_ques, self.conv_layers, self.self_attention_layer,
                               self.ffn_layer, ques_len, dropout,
                               num_blocks=self.encoder_layer_size, repeat=1)[1]

        x, S_bar, S_T = self.cqattention_layer([x_cont, x_ques, cont_len, ques_len])
        x = self.h2o_squeeze_layer(x)

        outputs = encoder_block(x, self.conv_layers2, self.self_attention_layer2,
                                self.ffn_layer2, cont_len, dropout,
                                num_blocks=self.output_layer_size, repeat=3)

        x_start = Concatenate()([outputs[1], outputs[2]])
        x_start = self.start_layer(x_start)
        x_start = Lambda(lambda x: tf.squeeze(x, axis=-1))(x_start)
        x_start = Lambda(lambda x: mask_logits(x[0], x[1], axis=0, time_dim=1))([x_start, cont_len])
        x_start = Lambda(lambda x: K.softmax(x), name='start')(x_start)

        x_end = Concatenate()([outputs[1], outputs[3]])
        x_end = self.end_layer(x_end)  # batch * seq_len * 1
        x_end = Lambda(lambda x: tf.squeeze(x, axis=-1))(x_end)
        x_end = Lambda(lambda x: mask_logits(x[0], x[1], axis=0, time_dim=1))([x_end, cont_len])
        x_end = Lambda(lambda x: K.softmax(x), name='end')(x_end)  # batch * seq_len

        return Model(inputs=[ques_input, cont_input], outputs=[x_start, x_end, S_bar, S_T])


def encoder_block_simple(x, conv_layers, ffn_layer, seq_len, dropout,
                         num_blocks, repeat=1, position=False):
    outputs = [x]
    for _ in range(repeat):
        x = outputs[-1]
        for j in range(num_blocks):
            if position:
                x = PositionEmbedding()(x)
            x = conv_block(x, conv_layers[j], dropout, j, num_blocks)
            x = ffn_block(x, ffn_layer[j], dropout, j, num_blocks)
        outputs.append(x)
    return outputs


class DependencyQANet:
    def __init__(self, vocab_size, embed_size, output_size, filters=128, num_heads=1,
                 ques_limit=50, dropout=0.1, num_blocks=1, num_convs=2,
                 embeddings=None, only_conv=False):
        self.ques_limit = ques_limit
        self.num_blocks = num_blocks
        self.num_convs = num_convs
        self.dropout = dropout
        self.only_conv = only_conv
        if embeddings is not None:
            embeddings = [embeddings]
        self.embed_layer = Embedding(
            vocab_size, embed_size, weights=embeddings, trainable=False)
        self.highway = Highway(filters, dropout=dropout, regularizer=regularizer)
        conv_layers = []
        self_attention_layer = []
        ffn_layer = []
        for i in range(num_blocks):
            conv_layers.append([])
            for j in range(num_convs):
                # conv_layers[i].append([
                #     DepthwiseConv2D((7, 1), padding='same', depth_multiplier=1, kernel_regularizer=regularizer),
                #     Conv2D(filters, 1, padding='same', kernel_regularizer=regularizer)])
                conv_layers[i].append(SeparableConv2D(
                    filters=filters, kernel_size=(7, 1), padding='same', activation='relu',
                    depthwise_regularizer=regularizer, pointwise_regularizer=regularizer))
            if not only_conv:
                self_attention_layer.append([
                    Conv1D(2 * filters, 1, kernel_regularizer=regularizer),  # weights for key and value
                    Conv1D(filters, 1, kernel_regularizer=regularizer),  # weights for query
                    MultiHeadAttention(filters, num_heads)])
            ffn_layer.append([Conv1D(filters, 1, activation='relu', kernel_regularizer=regularizer),
                              Conv1D(filters, 1, activation='linear', kernel_regularizer=regularizer)])
        self.conv_layers = conv_layers
        self.self_attention_layer = self_attention_layer
        self.ffn_layer = ffn_layer

        self.output_layer = Conv1D(output_size, 1, activation='linear', kernel_regularizer=regularizer)

    def build(self):
        dropout = self.dropout

        ques_input = Input((self.ques_limit,))

        # mask
        q_mask = Lambda(lambda x: tf.cast(x, tf.bool))(ques_input)
        ques_len = Lambda(lambda x: tf.expand_dims(tf.reduce_sum(tf.cast(x, tf.int32), axis=1), axis=1))(q_mask)

        # encoding each
        x_ques = self.embed_layer(ques_input)
        x_ques = self.highway(x_ques)
        if not self.only_conv:
            x_ques = encoder_block(x_ques, self.conv_layers, self.self_attention_layer,
                                   self.ffn_layer, ques_len, dropout,
                                   num_blocks=self.num_blocks, repeat=1)[1]
        else:
            x_ques = encoder_block_simple(x_ques, self.conv_layers,
                                          self.ffn_layer, ques_len, dropout,
                                          num_blocks=self.num_blocks, repeat=1)[1]

        y = self.output_layer(x_ques)  # batch * seq_len * output_size

        def mask_sequence(x, length):
            mask = tf.expand_dims(tf.sequence_mask(
                tf.squeeze(length, axis=1), maxlen=self.ques_limit, dtype=tf.float32), dim=2)
            return x * mask

        y = Lambda(lambda x: K.softmax(x), name='end')(y)  # batch * seq_len
        y = Lambda(lambda x: mask_sequence(x[0], x[1]))([y, ques_len])
        y = Masking(mask_value=0.)(y)

        return Model(inputs=ques_input, outputs=y)


class DependencyLSTM:
    def __init__(self, vocab_size, embed_size, output_size, hidden_size=128,
                 ques_limit=50, dropout=0.1, embeddings=None):
        self.ques_limit = ques_limit
        if embeddings is not None:
            embeddings = [embeddings]
        self.embed_layer = Embedding(
            vocab_size, embed_size, weights=embeddings, trainable=False, mask_zero=True)
        self.lstm = Bidirectional(LSTM(hidden_size, return_sequences=True))
        self.output_layer = Dense(output_size, activation='softmax')

    def build(self):
        ques_input = Input((self.ques_limit,))

        # encoding each
        x_ques = self.embed_layer(ques_input)
        x_ques = self.lstm(x_ques)

        y = self.output_layer(x_ques)  # batch * seq_len * output_size

        return Model(inputs=ques_input, outputs=y)
