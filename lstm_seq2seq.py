import csv
from collections import Counter

import spacy

import tensorflow as tf
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, concatenate, Lambda
import numpy as np


spacy_en = spacy.load('en_core_web_sm',
                      disable=['vectors', 'textcat', 'tagger', 'parser', 'ner'])


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

    index_to_token = ('<unk>', '<s>', '</s>', '<pad>') + ordered_tokens
    if len(index_to_token) > max_size:
        index_to_token = index_to_token[:max_size]
    indices = range(len(index_to_token))
    token_to_index = dict(zip(index_to_token, indices))
    return token_to_index, list(index_to_token)


with open('data/train-v2.0.txt') as f:
    data = [row for row in csv.reader(f, delimiter='\t')]

data = [[tokenizer(x[0]), tokenizer(x[1]), int(x[2]), int(x[3]), tokenizer(x[4])]
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
decoder_token_to_index = {'ignore': 0, 'start': 1, 'keep': 2}
decoder_index_to_token = ['ignore', 'start', 'keep']

spans = get_spans(contexts, char_starts, char_ends)
for i, spans in enumerate(spans):
    if spans[0] >= 0:
        start = spans[0]
        end = spans[1]
        decoder_target_data[i, start] = 1
        decoder_target_data[i, start + 1: end] = 2

decoder_target_data = decoder_target_data[:, :, None]

encoder_inputs = Input(shape=(max_encoder_seq_length,))
embedding = Embedding(len(token_to_index), latent_dim)
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(embedding(encoder_inputs))
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(max_decoder_seq_length,))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
# decoder_outputs, _, _ = decoder_lstm(embedding(decoder_inputs),
#                                      initial_state=encoder_states)
# decoder_outputs = decoder_dense(decoder_outputs)
states = encoder_states
embs = embedding(decoder_inputs)
embs = Lambda(lambda x: tf.transpose(x, ([1, 0, 2])))(embs)
expand_dims = Lambda(lambda x: tf.expand_dims(x, axis=0))
all_outputs = []
for emb in Lambda(lambda x: tf.unstack(x))(embs):
    outputs, state_h, state_c = decoder_lstm(expand_dims(emb), initial_state=states)
    states = [state_h, state_c]
    all_outputs.append(outputs)
decoder_outputs = Lambda(lambda x: concatenate(x, axis=0))(all_outputs)


model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0)
model.save('s2s.h5')


def decode_sequence(question_seq, context_seq):
    # Encode the input as state vectors.
    target_seq = model.predict([question_seq, context_seq], batch_size=1)

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    decoded_tokens = []
    for output_token in target_seq:
        # Sample a token
        sampled_token_index = np.argmax(output_token)
        sampled_char = decoder_index_to_token[sampled_token_index]
        decoded_tokens.append(sampled_char)
    decoded_sentence = ' '.join(decoded_tokens)

    return decoded_sentence


print(spans)
for seq_index in range(10):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    question_seq = encoder_input_data[seq_index]
    context_seq = decoder_input_data[seq_index]

    decoded_sentence = decode_sequence(question_seq, context_seq)
    print('-')
    print('Input sentence:', encoder_texts.data[seq_index])
    print('Decoded sentence:', decoded_sentence)
