import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import Model
from keras.layers import Input, LSTM, Dense, Embedding, Concatenate
from keras.engine.topology import Layer


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
