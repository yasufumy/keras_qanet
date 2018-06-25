import os
from argparse import ArgumentParser

import spacy

from models import SquadBaseline
from data import SquadReader, Iterator, SquadConverter, SquadTestConverter, Vocabulary,\
    load_squad_tokens
from trainer import SquadTrainer
from metrics import SquadMetric
from utils import evaluate


parser = ArgumentParser()
parser.add_argument('--epoch', default=100, type=int)
args = parser.parse_args()


spacy_en = spacy.load('en_core_web_sm',
                      disable=['vectors', 'textcat', 'tagger', 'parser', 'ner'])


def tokenizer(x): return [token.text.lower() for token in spacy_en(x) if not token.is_space]


if not os.path.exists('vocab.pkl'):
    squad_tokens = load_squad_tokens('./data/train-v2.0.txt')
    token_to_index, index_to_token = Vocabulary.build(
        squad_tokens, 5, 3000, ('<pad>', '<unk>'), 'vocab.pkl')

else:
    token_to_index, index_to_token = Vocabulary.load('vocab.pkl')

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
trainer.run()
model.save('s2s.h5')

metric = SquadMetric()
dataset = SquadReader('data/dev-v2.0.txt')
converter = SquadTestConverter(token_to_index, 1, '<pad>', 3)
dev_generator = Iterator(dataset, batch_size, converter, False, False)
em_score, f1_score = evaluate(inference, dev_generator, metric, 1, 2, index_to_token)
print('EM: {}, F1: {}'.format(em_score, f1_score))
