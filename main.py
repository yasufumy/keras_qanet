import os
import csv
import pickle
from argparse import ArgumentParser

import spacy

from models import SquadBaseline
from data import SquadReader, Iterator, SquadConverter, SquadTestConverter, make_vocab
from trainer import SquadTrainer
from metrics import SquadMetric


parser = ArgumentParser()
parser.add_argument('--epoch', default=100, type=int)
args = parser.parse_args()


spacy_en = spacy.load('en_core_web_sm',
                      disable=['vectors', 'textcat', 'tagger', 'parser', 'ner'])


def tokenizer(x): return [token for token in spacy_en(x) if not token.is_space]


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
trainer = SquadTrainer(model, train_generator, epochs)
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
