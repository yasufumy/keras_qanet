import pickle
import csv
import math
import linecache
from collections import Counter
from itertools import takewhile

import numpy as np
import spacy

from utils import get_spans


def make_vocab(tokens, min_count, max_vocab_size,
               speicial_tokens=('<pad>', '<unk>', '<s>', '</s>')):
    counter = Counter(tokens)
    ordered_tokens, _ = zip(*takewhile(lambda x: x[1] >= min_count,
                                       counter.most_common()))
    index_to_token = speicial_tokens + ordered_tokens
    if len(index_to_token) > max_vocab_size:
        index_to_token = index_to_token[:max_vocab_size]
    indices = range(len(index_to_token))
    token_to_index = dict(zip(index_to_token, indices))
    return token_to_index, list(index_to_token)


def load_squad_tokens(filename, tokenizer):
    with open(filename) as f:
        data = [row for row in csv.reader(f, delimiter='\t')]
    print(data)
    data = [[tokenizer(x[0]), tokenizer(x[1])] for x in data]
    print(data)
    contexts, questions = zip(*data)
    tokens = (token for tokens in contexts + questions for token in tokens)
    return tokens


class Vocabulary:
    @staticmethod
    def build(tokens, min_count, max_vocab_size, speicial_tokens, savefile=None):
        token_to_index, index_to_token = make_vocab(tokens, min_count, max_vocab_size, speicial_tokens)
        if savefile is not None:
            with open(savefile, mode='wb') as f:
                pickle.dump((token_to_index, index_to_token), f)
        return token_to_index, index_to_token

    @staticmethod
    def load(filename):
        with open(filename, mode='rb') as f:
            token_to_index, index_to_token = pickle.load(f)
        return token_to_index, index_to_token


class SquadReader:
    def __init__(self, filename):
        self._filename = filename
        with open(filename) as f:
            self._total_data = len(f.readlines()) - 1

    def __getitem__(self, i):
        if isinstance(i, slice):
            lines = [
                linecache.getline(self._filename, i + 1)
                for i in range(i.start or 0, min(i.stop, len(self)), i.step or 1)]
            data = [row for row in csv.reader(lines, delimiter='\t')]
        else:
            if i > self._total_data:
                raise IndexError('Invalid Index')
            lines = [linecache.getline(self._filename, i + 1)]
            data = next(csv.reader(lines, delimiter='\t'))
        return data

    def __len__(self):
        return self._total_data


class Iterator:
    def __init__(self, dataset, batch_size, converter, repeat=True, shuffle=True):
        self._dataset = dataset
        self._batch_size = batch_size
        self._converter = converter
        self._repeat = repeat
        self._shuffle = shuffle
        self._epoch = 0

        self.reset()

    def reset(self):
        self._current_position = 0
        if self._shuffle:
            self._order = np.random.permutation(len(self._dataset))
        else:
            self._order = None

    def __len__(self):
        return math.ceil(len(self._dataset) / self._batch_size)

    def __iter__(self):
        return self

    def __next__(self):
        if not self._repeat and self._epoch > 0:
            raise StopIteration
        i = self._current_position
        i_end = i + self._batch_size
        N = len(self._dataset)

        if self._order is not None:
            batch = [self._dataset[index] for index in self._order[i:i_end]]
        else:
            batch = self._dataset[i:i_end]

        if i_end >= N:
            if self._repeat:
                rest = i_end - N
                np.random.shuffle(self._order)
                if rest > 0:
                    batch.extend(
                        [self._dataset[index] for index in self._order[:rest]])
                self._current_position = rest
            else:
                self._current_position = 0

            self._epoch += 1
        else:
            self._current_position = i_end

        return self._converter(batch)


class SquadConverter:
    def __init__(self, token_to_index, unk_index, pad_token, categories):
        spacy_en = spacy.load(
            'en_core_web_sm', disable=['vectors', 'textcat', 'tagger', 'parser', 'ner'])

        def tokenizer(x):
            return [token for token in spacy_en(x) if not token.is_space]

        self._tokenizer = tokenizer
        self._token_to_index = token_to_index
        self._unk_index = unk_index
        self._pad_token = pad_token
        self._categories = categories

    def __call__(self, batch):
        contexts, questions, starts, ends, answers = zip(*batch)

        contexts = [self._tokenizer(context) for context in contexts]
        questions = [self._tokenizer(question) for question in questions]
        starts = [int(start) for start in starts]
        ends = [int(end) for end in ends]
        spans = get_spans(contexts, starts, ends)

        context_batch = self._process_text(contexts)
        question_batch = self._process_text(questions)
        output_span_batch = np.zeros(context_batch.shape, dtype=np.int32)
        input_span_batch = np.zeros(context_batch.shape + (self._categories,))
        for i, span in enumerate(spans):
            if span[0] >= 0:
                start, end = span
                output_span_batch[i, start] = 1
                output_span_batch[i, start + 1:end + 1] = 2
                input_span_batch[i, :start + 1, 0] = 1.
                if start + 1 < input_span_batch.shape[1]:
                    input_span_batch[i,  start + 1, 1] = 1.
                if start + 2 < input_span_batch.shape[1]:
                    input_span_batch[i,  start + 2: end + 1, 2] = 1.
        return [question_batch, context_batch, input_span_batch], output_span_batch[:, :, None]

    def _process_text(self, texts):
        texts = [[token.text for token in text] for text in texts]
        max_length = max(len(x) for x in texts)
        texts = [x + [self._pad_token] * (max_length - len(x)) for x in texts]
        return np.array([
            [self._token_to_index.get(token, self._unk_index) for token in text]
            for text in texts], dtype=np.int32)


class SquadTestConverter(SquadConverter):
    def __call__(self, batch):
        contexts, questions, _, _, answers = zip(*batch)
        contexts = [self._tokenizer(context) for context in contexts]
        questions = [self._tokenizer(question) for question in questions]
        context_batch = self._process_text(contexts)
        question_batch = self._process_text(questions)
        return question_batch, context_batch, answers
