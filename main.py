import os
import csv
import string
import math
import linecache
import pickle
from collections import Counter
from argparse import ArgumentParser

import spacy

import numpy as np

from models import SquadBaseline
from data import SquadReader, Iterator, SquadConverter, SquadTestConverter, make_vocab


parser = ArgumentParser()
parser.add_argument('--epoch', default=100, type=int)
args = parser.parse_args()


spacy_en = spacy.load('en_core_web_sm',
                      disable=['vectors', 'textcat', 'tagger', 'parser', 'ner'])


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


class SquadTestGenerator:
    def __init__(self, filename, batch_size):
        self._filename = filename
        with open(filename) as f:
            self._total_data = len(f.readlines()) - 1
        self._batch_size = batch_size
        indices = range(self._total_data)
        self._indices = [indices[i:i + self._batch_size] for i in range(0, self._total_data, self._batch_size)]

    def __len__(self):
        return int(math.ceil(self._total_data / float(self._batch_size)))

    def __iter__(self):
        for indices in self._indices:
            contexts, questions, _, _, answers = zip(*csv.reader(
                [linecache.getline(self._filename, i + 1) for i in indices], delimiter='\t'))

            contexts = [tokenizer(x) for x in contexts]
            questions = [tokenizer(x) for x in questions]
            question_batch = TextData(questions).getattr('text').padding('<pad>').to_array(token_to_index, 0).data
            context_batch = TextData(contexts).getattr('text').padding('<pad>').to_array(token_to_index, 0).data

            yield question_batch, context_batch, answers


if not os.path.exists('vocab.pkl'):
    with open('data/squad_train_v2.0/train-v2.0.txt') as f:
        data = [row for row in csv.reader(f, delimiter='\t')]
    data = [[tokenizer(x[0]), tokenizer(x[1]), int(x[2]), int(x[3]), x[4]]
            for x in data]
    contexts, questions, char_starts, char_ends, answers = zip(*data)
    tokens = (token.text for tokens in contexts + questions for token in tokens)
    token_to_index, index_to_token = make_vocab(tokens, 30000)
    with open('vocab.pkl', mode='wb') as f:
        pickle.dump((token_to_index, index_to_token), f)
else:
    with open('vocab.pkl', mode='rb') as f:
        token_to_index, index_to_token = pickle.load(f)

batch_size = 256  # Batch size for training.
epochs = args.epoch  # Number of epochs to train for.
latent_dim = 128  # Latent dimensionality of the encoding space.
num_encoder_tokens = len(token_to_index)
num_decoder_tokens = 3


print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)

decoder_token_to_index = {'ignore': 0, 'start': 1, 'keep': 2}
decoder_index_to_token = ['ignore', 'start', 'keep']

model, inference = SquadBaseline(len(token_to_index), latent_dim, latent_dim, 3).build()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
dataset = SquadReader('data/train-v2.0.txt')
converter = SquadConverter(token_to_index, 1, '<pad>', 3)
train_generator = Iterator(dataset, batch_size, converter)
model.fit_generator(
    generator=train_generator, steps_per_epoch=len(train_generator), epochs=epochs)
model.save('s2s.h5')

metric = SquadMetric()
dataset = SquadReader('data/dev-v2.0.txt')
converter = SquadTestConverter(token_to_index, 1, '<pad>', 3)
dev_generator = Iterator(dataset, batch_size, converter, False, False)
for question, context, answer in dev_generator:
    decoded_sentences = inference(question, context)
    for i, sent in enumerate(zip(*decoded_sentences)):
        indices = [j for j, y in enumerate(sent) if y == 1 or y == 2]
        prediction = ' '.join(index_to_token[context[i][j]] for j in indices)
        metric(prediction, answer[i])
print('EM: {}, F1: {}'.format(*metric.get_metric()))
