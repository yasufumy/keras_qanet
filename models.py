import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import Model
from keras.layers import Input, LSTM, Dense, Embedding, Concatenate, Lambda, \
    BatchNormalization, DepthwiseConv2D, Conv1D, Conv2D
from keras.engine.topology import Layer

from layers import MultiHeadAttention, PositionEmbedding, ContextQueryAttention


class DotAttentionLayer(Layer):
    def call(self, inputs):
        keys, query = inputs
        if len(K.int_shape(query)) == 2:
            # when query is a vector
            query = tf.expand_dims(query, dim=1)
        scores = tf.matmul(query, keys, transpose_b=True)
        # scores_mask = tf.expand_dims(tf.sequence_mask(
        #     lengths=tf.to_int32(tf.squeeze(lengths, axis=1)),
        #     maxlen=tf.to_int32(tf.shape(scores)[1]),
        #     dtype=tf.float32), dim=2)
        # scores = scores * scores_mask + (1. - scores_mask) * tf.float32.min
        weights = K.softmax(scores, axis=2)
        return tf.matmul(weights, keys)

    def compute_mask(self, inputs, mask=None):
        # just feeding query's mask
        if mask is not None:
            return mask[1]
        else:
            return None


class SquadBaseline:
    def __init__(self, vocab_size, embed_size, hidden_size, categories):
        self.embed_layer = Embedding(vocab_size, embed_size, mask_zero=True)
        self.encode_layer = LSTM(hidden_size, return_state=True, return_sequences=True)
        self.decode_layer = LSTM(hidden_size, return_state=True, return_sequences=True)
        self.output_layer = Dense(categories, activation='softmax')
        self.attention_layer = DotAttentionLayer()

        self._hidden_size = hidden_size
        self._categories = categories

    def build(self):
        # placeholders
        encoder_inputs = Input(shape=(None,))
        decoder_inputs = Input(shape=(None,))
        last_outputs = Input(shape=(None, self._categories))
        # utility
        concat = Concatenate(axis=-1)

        # graph
        encoder_outputs, *encoder_states = self.encode_layer(
            self.embed_layer(encoder_inputs))

        decoder_outputs, _, _ = self.decode_layer(
            concat([self.embed_layer(decoder_inputs), last_outputs]))
        attention_outputs = self.attention_layer([encoder_outputs, decoder_outputs])
        model_outputs = self.output_layer(concat([decoder_outputs, attention_outputs]))

        model = Model([encoder_inputs, decoder_inputs, last_outputs], model_outputs)

        # inference
        encoder_model = Model(encoder_inputs, [encoder_outputs] + encoder_states)
        decoder_states_inputs = [Input(shape=(self._hidden_size,)),
                                 Input(shape=(self._hidden_size,))]
        encoded_inputs = Input(shape=(None, self._hidden_size))
        decoder_outputs, *decoder_states = self.decode_layer(
            concat([self.embed_layer(decoder_inputs), last_outputs]),
            initial_state=decoder_states_inputs)
        attention_outputs = self.attention_layer([encoded_inputs, decoder_outputs])
        model_outputs = self.output_layer(concat([decoder_outputs, attention_outputs]))
        decoder_model = Model(
            [decoder_inputs, last_outputs, encoded_inputs] + decoder_states_inputs,
            [model_outputs] + decoder_states)

        return model, self._build_inference(encoder_model, decoder_model, self._categories)

    @staticmethod
    def _build_inference(encoder_model, decoder_model, categories):
        def inference(question, context):
            batch_size, _ = question.shape
            encoder_outputs, *states = encoder_model.predict([question])

            decoded_tokens = []
            last_tokens = np.zeros((batch_size, 1, categories))
            for tokens in np.transpose(context, [1, 0]):
                outputs, *states = decoder_model.predict(
                    [tokens, last_tokens, encoder_outputs] + states)
                outputs = np.squeeze(outputs)
                sampled_tokens = np.argmax(outputs, axis=1).tolist()
                decoded_tokens.append(sampled_tokens)

                last_tokens = np.identity(categories)[sampled_tokens][:, None, :]

            return decoded_tokens
        return inference


def encoder_block(x, conv_layers, attention_layer, ffn_layer, seq_len,
                  num_blocks, repeat=1):

    def conv_block(x, conv_layers):
        n_conv = len(conv_layers)
        x = Lambda(lambda x: tf.expand_dims(x, axis=2))(x)
        for i in range(n_conv):
            residual = x
            x = BatchNormalization()(x)
            x = conv_layers[i](x)
            x = x + residual
        x = Lambda(lambda x: tf.squeeze(x, axis=2))(x)
        return x

    def attention_block(x, attention_layer, seq_len):
        residual = x
        x = BatchNormalization()(x)
        key_and_value = attention_layer[0](x)
        query = attention_layer[1](x)
        x = attention_layer[2]([key_and_value, query, seq_len])
        return x + residual

    def ffn_block(self, x, ffn_layer):
        residual = x
        x = BatchNormalization()(x)
        x = ffn_layer[0](x)
        x = ffn_layer[1](x)
        return x + residual

    outputs = [x]
    for i in range(repeat):
        x = outputs[-1]
        for j in range(num_blocks):
            x = PositionEmbedding()(x)
            x = conv_block(x, conv_layers)
            x = attention_block(x, attention_layer)
            x = ffn_block(x, ffn_layer)
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


class LightQANet:
    def __init__(self, vocab_size, embed_size, filters=128, cont_limit=400, ques_limit=50):
        self.cont_limit = cont_limit
        self.ques_limit = ques_limit
        self.embed_layer = Embedding(vocab_size, embed_size)
        conv_layers = []
        for i in range(2):
            conv_layers.append([
                DepthwiseConv2D((7, 1), padding='same', depth_multiplier=1),
                Conv2D(filters, 1, padding='same')])
        self.conv_layers = conv_layers
        self.self_attention_layer = [
            Conv1D(2 * filters, 1),  # weights for key and value
            Conv1D(filters, 1),  # weights for query
            MultiHeadAttention()]
        self.ffn_layer = [Conv1D(filters, 1, activation='relu'),
                          Conv1D(filters, 1, activation='linear')]
        self.cqattention_layer = ContextQueryAttention(512, cont_limit, ques_limit)
        conv_layers = []
        self_attention_layer = []
        ffn_layer = []
        for i in range(3):
            conv_layers.append([])
            for j in range(2):
                conv_layers[i].append([
                    DepthwiseConv2D((7, 1), padding='same', depth_multiplier=1),
                    Conv2D(filters, 1, padding='same')])
            self_attention_layer.append([
                Conv1D(2 * filters, 1),
                Conv1D(filters, 1),
                MultiHeadAttention()])
            ffn_layer.append([Conv1D(filters, 1, activation='relu'),
                              Conv1D(filters, 1, activation='linear')])
        self.conv_layers2 = conv_layers
        self.self_attention_layer2 = self_attention_layer
        self.ffn_layer2 = ffn_layer

        self.start_layer = Conv1D(1, 1, activation='linear')
        self.end_layer = Conv1D(1, 1, activation='linear')

    def build(self):
        cont_input = Input((self.cont_limit,))
        ques_input = Input((self.ques_limit,))

        # mask
        c_mask = Lambda(lambda x: tf.cast(x, tf.bool))(cont_input)
        q_mask = Lambda(lambda x: tf.cast(x, tf.bool))(ques_input)
        cont_len = Lambda(lambda x: tf.expand_dims(tf.reduce_sum(tf.cast(x, tf.int32), axis=1), axis=1))(c_mask)
        ques_len = Lambda(lambda x: tf.expand_dims(tf.reduce_sum(tf.cast(x, tf.int32), axis=1), axis=1))(q_mask)

        # encoding each
        x_cont = self.embed_layer(cont_input)
        x_cont = encoder_block(x_cont, self.conv_layers, self.self_attention_layer,
                               self.ffn_layer, cont_len, num_blocks=1, repeat=1)[1]

        x_ques = self.embed_layer(ques_input)
        x_ques = encoder_block(x_ques, self.conv_layers, self.self_attention_layer,
                               self.ffn_layer, ques_len, num_blocks=1, repeat=1)[1]

        x = self.cqattention_layer([x_cont, x_ques, cont_len, ques_len])

        outputs = encoder_block(x, self.conv_layers2, self.self_attention_layer2,
                                self.ffn_layer2, cont_len, num_blocks=3, repeat=3)

        x_start = Concatenate()([outputs[1], outputs[2]])
        x_start = self.start_layer(x_start)
        x_start = Lambda(lambda x: tf.squeeze(x, axis=-1))(x_start)
        x_start = Lambda(lambda x: mask_logits(x[0], x[1], axis=0, time_dim=1))([x_start, cont_len])
        x_start = Lambda(lambda x: K.softmax(x))(x_start)

        x_end = Concatenate()([outputs[1], outputs[3]])
        x_end = self.start_layer(x_end)
        x_end = Lambda(lambda x: tf.squeeze(x, axis=-1))(x_end)
        x_end = Lambda(lambda x: mask_logits(x[0], x[1], axis=0, time_dim=1))([x_end, cont_len])
        x_end = Lambda(lambda x: K.softmax(x))(x_end)

        return Model(inputs=[cont_input, ques_input], outputs=[x_start, x_end])
