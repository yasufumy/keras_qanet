import csv
import string
from collections import Counter

import spacy

import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, Concatenate
from keras.engine.topology import Layer
import numpy as np


spacy_en = spacy.load('en_core_web_sm',
                      disable=['vectors', 'textcat', 'tagger', 'parser', 'ner'])


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


def normalize_answer(text):
    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(char for char in text if char not in exclude)

    return white_space_fix(remove_punc(str.lower(text)))


def f1_score(prediction, ground_truth):
    if prediction == ground_truth == '':
        return 1
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1. * num_same / len(prediction_tokens)
    recall = 1. * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_groud_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_groud_truths.append(score)
    return max(scores_for_groud_truths)


class SquadMetric:
    def __init__(self):
        self._total_em = 0.
        self._total_f1 = 0.
        self._count = 0

    def __call__(self, best_span_string, answer_string):
        em = metric_max_over_ground_truths(
            exact_match_score, best_span_string, [answer_string])
        f1 = metric_max_over_ground_truths(
            f1_score, best_span_string, [answer_string])
        self._total_em += em
        self._total_f1 += f1
        self._count += 1

    def get_metric(self, reset=False):
        em = self._total_em / self._count if self._count > 0 else 0
        f1 = self._total_f1 / self._count if self._count > 0 else 0
        if reset:
            self._total_em = 0.
            self._total_f1 = 0.
            self._count = 0
        return em, f1


def char_span_to_token_span(token_offsets, char_start, char_end):
    if char_start < 0:
        return (-1, -1), False

    error = False

    start_index = 0
    while start_index < len(token_offsets) and token_offsets[start_index][0] < char_start:
        start_index += 1
    if token_offsets[start_index][0] != char_start:
        error = True

    end_index = 0
    while end_index < len(token_offsets) and token_offsets[end_index][1] < char_end:
        end_index += 1
    if token_offsets[end_index][1] != char_end:
        error = True
    return (start_index, end_index), error


class TextData:
    def __init__(self, data):
        self.data = data

    def to_array(self, token_to_index, unk_index, dtype=np.int32):
        self.data = np.array([[token_to_index.get(token, unk_index)
                               for token in x] for x in self.data]).astype(np.int32)
        return self

    def reshape(self, shape):
        self.data = self.data.reshape(shape)
        return self

    def getattr(self, attr):
        self.data = [[getattr(token, attr) for token in x] for x in self.data]
        return self

    def padding(self, pad_token):
        max_length = self.max_length
        self.data = [x + [pad_token] * (max_length - len(x)) for x in self.data]
        return self

    @property
    def max_length(self):
        return max(len(x) for x in self.data)

    def __len__(self):
        return len(self.data)


def tokenizer(x): return [token for token in spacy_en(x) if not token.is_space]


def get_spans(contexts, starts, ends):
    spans = []
    for context, start, end in zip(contexts, starts, ends):
        context_offsets = [(token.idx, token.idx + len(token.text)) for token in context]
        span, error = char_span_to_token_span(context_offsets, start, end)
        if error:
            raise Exception('Failed')
        spans.append(span)

    return spans


def make_vocab(tokens, max_size):
    counter = Counter(tokens)
    ordered_tokens, _ = zip(*counter.most_common())

    index_to_token = ('<pad>', '<unk>', '<s>', '</s>') + ordered_tokens
    if len(index_to_token) > max_size:
        index_to_token = index_to_token[: max_size]
    indices = range(len(index_to_token))
    token_to_index = dict(zip(index_to_token, indices))
    return token_to_index, list(index_to_token)


with open('data/train-v2.0.txt') as f:
    data = [row for row in csv.reader(f, delimiter='\t')]

data = [[tokenizer(x[0]), tokenizer(x[1]), int(x[2]), int(x[3]), x[4]]
        for x in data[:10]]
contexts, questions, char_starts, char_ends, answers = zip(*data)
tokens = (token.text for tokens in contexts + questions for token in tokens)
token_to_index, index_to_token = make_vocab(tokens, 30000)

encoder_texts = TextData(questions).getattr('text').padding('<pad>').to_array(token_to_index, 0)
decoder_texts = TextData(contexts).getattr('text').padding('<pad>').to_array(token_to_index, 0)

batch_size = 1  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_encoder_tokens = len(token_to_index)
num_decoder_tokens = 3
max_encoder_seq_length = encoder_texts.max_length
max_decoder_seq_length = decoder_texts.max_length


print('Number of samples:', len(encoder_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

encoder_input_data = encoder_texts.data
decoder_input_data = decoder_texts.data
decoder_target_data = np.zeros(decoder_texts.data.shape, dtype=np.int32)
decoder_input_data2 = np.zeros(decoder_texts.data.shape + (3,))
decoder_token_to_index = {'ignore': 0, 'start': 1, 'keep': 2}
decoder_index_to_token = ['ignore', 'start', 'keep']

spans = get_spans(contexts, char_starts, char_ends)
for i, spans in enumerate(spans):
    if spans[0] >= 0:
        start = spans[0]
        end = spans[1]
        decoder_target_data[i, start] = 1
        decoder_target_data[i, start + 1: end + 1] = 2
        if i < len(spans):
            decoder_input_data2[i + 1, start, 1] = 1.
            decoder_input_data2[i + 1, start + 1: end + 1, 2] = 1.

decoder_target_data = decoder_target_data[:, :, None]

encoder_inputs = Input(shape=(None,))
embedding = Embedding(len(token_to_index), latent_dim, mask_zero=True)
encoder = LSTM(latent_dim, return_state=True, return_sequences=True,
               batch_input_shape=(None, max_encoder_seq_length, latent_dim))
encoder_outputs, state_h, state_c = encoder(embedding(encoder_inputs))
encoder_states = [state_h, state_c]


decoder_inputs = Input(shape=(None,))
decoder_inputs2 = Input(shape=(None, 3))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True,
                    batch_input_shape=(None, max_decoder_seq_length, latent_dim))
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
concat = Concatenate(axis=-1)
decoder_outputs, _, _ = decoder_lstm(concat([embedding(decoder_inputs), decoder_inputs2]),
                                     initial_state=encoder_states)
attention = DotAttentionLayer()
attention_outputs = attention([encoder_outputs, decoder_outputs])
decoder_outputs = decoder_dense(concat([decoder_outputs, attention_outputs]))


model = Model([encoder_inputs, decoder_inputs, decoder_inputs2], decoder_outputs)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data, decoder_input_data2],
          decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0)
model.save('s2s.h5')


encoder_model = Model(encoder_inputs, [encoder_outputs] + encoder_states)

# input placeholder
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
encoder_outputs_inputs = Input(shape=(None, latent_dim))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
# feeding lstm
decoder_outputs, state_h, state_c = decoder_lstm(
    concat([embedding(decoder_inputs), decoder_inputs2]), initial_state=decoder_states_inputs)
attention_outputs = attention([encoder_outputs_inputs, decoder_outputs])
# model outputs
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(concat([decoder_outputs, attention_outputs]))
decoder_model = Model(
    [decoder_inputs, decoder_inputs2, encoder_outputs_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)


def decode_sequence(question_seq, context_seq):
    # Encode the input as state vectors.
    encoder_outputs, *states_value = encoder_model.predict([question_seq], batch_size=1)

    decoded_tokens = []
    action = np.zeros((1, 1, 3))
    for token in np.transpose(context_seq, [1, 0]):
        output_tokens, h, c = decoder_model.predict(
            [token, action, encoder_outputs] + states_value)
        sampled_token_index = np.argmax(output_tokens)
        sampled_char = decoder_index_to_token[sampled_token_index]
        decoded_tokens.append(sampled_char)

        action = np.zeros((1, 1, 3))
        action[0, 0, sampled_token_index] = 1.

        states_value = [h, c]

    return decoded_tokens


metric = SquadMetric()
for seq_index in range(10):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    question_seq = encoder_input_data[seq_index][None]
    context_seq = decoder_input_data[seq_index][None]

    decoded_sentence = decode_sequence(question_seq, context_seq)
    indices = [i for i, y in enumerate(decoded_sentence) if y == 'start' or y == 'keep']
    prediction = ' '.join(index_to_token[decoder_texts.data[seq_index][i]] for i in indices)
    answer = answers[seq_index]
    metric(prediction, answer)
    print(f'prediction: {prediction}, answer: {answer}')
print(metric.get_metric())
