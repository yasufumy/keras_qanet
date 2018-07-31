import tensorflow as tf
from keras import backend as K
from keras import Model
from keras.regularizers import l2
from keras.initializers import VarianceScaling
from keras.layers import Input, Embedding, Concatenate, Lambda, \
    BatchNormalization, DepthwiseConv2D, Conv1D, Conv2D, Dropout, Masking, \
    LSTM, Bidirectional, Dense, SeparableConv1D

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
    for i in range(n_conv):
        residual = x
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        x = conv_layers[i](x)
        x = LayerDropout(dropout * (sub_layer / last_layer))([x, residual])
    return x


def attention_block(x, attention_layer, seq_len, dropout=0., sub_layer=1., last_layer=1.):
    residual = x
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    # key_and_value = attention_layer[0](x)
    # query = attention_layer[1](x)
    x = attention_layer[0]([x, x, x, seq_len])
    return LayerDropout(dropout * (sub_layer / last_layer))([x, residual])


def ffn_block(x, ffn_layer, dropout=0., sub_layer=1., last_layer=1.):
    residual = x
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    x = ffn_layer[0](x)
    x = ffn_layer[1](x)
    return LayerDropout(dropout * (sub_layer / last_layer))([x, residual])


def encoder_block(x, conv_layers, attention_layer, ffn_layer, seq_len, dropout,
                  num_blocks):
    num_convs = len(conv_layers[0])
    total_layer = (2 + num_convs) * num_blocks
    sub_layer = 1

    for i in range(num_blocks):
        x = PositionEmbedding()(x)
        for j in range(num_convs):
            residual = x
            x = BatchNormalization()(x)
            if sub_layer % 2 == 0:
                x = Dropout(dropout)(x)
            x = conv_layers[i][j](x)
            x = LayerDropout(dropout * (sub_layer / total_layer))([x, residual])
            sub_layer += 1
        residual = x
        x = BatchNormalization()(x)
        if sub_layer % 2 == 0:
            x = Dropout(dropout)(x)
        x = attention_layer[i]([x, x, x, seq_len])
        x = LayerDropout(dropout * (sub_layer / total_layer))([x, residual])
        residual = x
        x = BatchNormalization()(x)
        if sub_layer % 2 == 0:
            x = Dropout(dropout)(x)
        x = ffn_layer[i][0](x)
        x = ffn_layer[i][1](x)
        x = LayerDropout(dropout * (sub_layer / total_layer))([x, residual])
        sub_layer += 1
    return x


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
                conv_layers[i].append([
                    DepthwiseConv2D((7, 1), padding='same', depth_multiplier=1,
                                    kernel_regularizer=regularizer, activation='relu'),
                    Conv2D(filters, 1, padding='same', kernel_regularizer=regularizer)
                ])
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
                conv_layers[i].append([
                    DepthwiseConv2D((5, 1), padding='same', depth_multiplier=1, kernel_regularizer=regularizer),
                    Conv2D(filters, 1, padding='same', kernel_regularizer=regularizer)])
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
        # (batch, 1)
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

        def mask_sequence(x, mask, mask_value=tf.float32.min, axis=1):
            # x: (batch, cont_len)
            maxlen = x.shape.as_list()[-1]
            # mask: (batch, cont_len)
            mask = tf.squeeze(tf.sequence_mask(mask, maxlen=maxlen, dtype=tf.float32), axis=1)
            return x + mask_value * (1 - mask)

        x_start = Concatenate()([outputs[1], outputs[2]])
        x_start = self.start_layer(x_start)
        x_start = Lambda(lambda x: tf.squeeze(x, axis=-1))(x_start)
        x_start = Lambda(lambda x: mask_sequence(x[0], x[1]))([x_start, cont_len])
        x_start = Lambda(lambda x: K.softmax(x), name='start')(x_start)

        x_end = Concatenate()([outputs[1], outputs[3]])
        x_end = self.end_layer(x_end)  # batch * seq_len * 1
        x_end = Lambda(lambda x: tf.squeeze(x, axis=-1))(x_end)
        x_end = Lambda(lambda x: mask_sequence(x[0], x[1]))([x_end, cont_len])
        x_end = Lambda(lambda x: K.softmax(x), name='end')(x_end)  # batch * seq_len

        return Model(inputs=[ques_input, cont_input], outputs=[x_start, x_end, S_bar, S_T])


class Encoder:
    def __init__(self, filters, kernel_size, num_blocks, num_convs, num_heads,
                 dropout, regularizer, **kwargs):
        conv_layers = []
        attention_layers = []
        feedforward_layers = []
        for i in range(num_blocks):
            conv_layers.append([])
            for j in range(num_convs):
                conv_layers[i].append(
                    SeparableConv1D(filters, 7, padding='same', depthwise_regularizer=regularizer,
                                    pointwise_regularizer=regularizer, activation='relu',
                                    bias_regularizer=regularizer, activity_regularizer=regularizer))
            attention_layers.append(
                MultiHeadAttention(filters, num_heads, dropout, regularizer))
            feedforward_layers.append([Conv1D(filters, 1, activation='relu', kernel_regularizer=regularizer),
                                       Conv1D(filters, 1, activation='linear', kernel_regularizer=regularizer)])

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
            for j in range(num_convs):
                residual = x
                x = BatchNormalization()(x)
                if sub_layer % 2 == 0:
                    x = Dropout(dropout)(x)
                x = conv_layers[i][j](x)
                x = LayerDropout(dropout * (sub_layer / total_layer))([x, residual])
                sub_layer += 1
            residual = x
            x = BatchNormalization()(x)
            if sub_layer % 2 == 0:
                x = Dropout(dropout)(x)
            x = attention_layers[i]([x, x, x, seq_len])
            x = LayerDropout(dropout * (sub_layer / total_layer))([x, residual])
            residual = x
            x = BatchNormalization()(x)
            if sub_layer % 2 == 0:
                x = Dropout(dropout)(x)
            x = feedforward_layers[i][0](x)
            x = feedforward_layers[i][1](x)
            x = LayerDropout(dropout * (sub_layer / total_layer))([x, residual])
            sub_layer += 1
        return x


class DependencyQANet:
    def __init__(self, vocab_size, embed_size, output_size, filters=128, num_heads=1,
                 ques_limit=50, dropout=0.1, num_blocks=1, num_convs=2, embeddings=None):
        self.ques_limit = ques_limit
        self.num_blocks = num_blocks
        self.num_convs = num_convs
        self.dropout = dropout
        if embeddings is not None:
            embeddings = [embeddings]
        self.embed_layer = Embedding(
            vocab_size, embed_size, weights=embeddings, trainable=False)
        self.highway = Highway(embed_size, 2, regularizer=regularizer, dropout=dropout)
        self.projection = Conv1D(filters, 1, kernel_regularizer=regularizer, activation='linear')
        # conv_layers = []
        # self_attention_layer = []
        # ffn_layer = []
        # for i in range(num_blocks):
        #     conv_layers.append([])
        #     for j in range(num_convs):
        #         conv_layers[i].append(
        #             SeparableConv1D(filters, 7, padding='same', depthwise_regularizer=regularizer,
        #                             pointwise_regularizer=regularizer, activation='relu',
        #                             bias_regularizer=regularizer, activity_regularizer=regularizer))
        #     self_attention_layer.append(
        #         MultiHeadAttention(filters, num_heads, dropout, regularizer))
        #     ffn_layer.append([Conv1D(filters, 1, activation='relu', kernel_regularizer=regularizer),
        #                       Conv1D(filters, 1, activation='linear', kernel_regularizer=regularizer)])
        # self.conv_layers = conv_layers
        # self.self_attention_layer = self_attention_layer
        # self.ffn_layer = ffn_layer
        self.encoder = Encoder(filters, 7, num_blocks, num_convs, num_heads,
                               dropout, regularizer)

        self.output_layer = Conv1D(output_size, 1, activation='linear', kernel_regularizer=regularizer)

    def build(self):
        # dropout = self.dropout

        ques_input = Input((self.ques_limit,))

        # mask
        q_mask = Lambda(lambda x: tf.cast(x, tf.bool))(ques_input)
        # ques_len: (batch, 1)
        ques_len = Lambda(lambda x: tf.expand_dims(tf.reduce_sum(tf.cast(x, tf.int32), axis=1), axis=1))(q_mask)

        # encoding each
        x_ques = self.embed_layer(ques_input)
        x_ques = self.highway(x_ques)
        x_ques = self.projection(x_ques)
        x_ques = self.encoder(x_ques, ques_len)

        y = self.output_layer(x_ques)  # batch * seq_len * output_size

        def mask_sequence(x, mask):
            # x: (batch, seq_len, output_size)
            # mask: (batch, 1)
            mask = tf.transpose(
                tf.sequence_mask(mask, maxlen=self.ques_limit, dtype=tf.float32),
                [0, 2, 1])
            # mask: (batch, seq_len, 1)
            return x * mask

        y = Lambda(lambda x: tf.nn.softmax(x, axis=-1), name='end')(y)  # batch * seq_len * output_size
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
