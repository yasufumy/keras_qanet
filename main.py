import os
from argparse import ArgumentParser

import spacy

from models import LightQANet
from data import SquadReader, Iterator, SquadConverter, Vocabulary,\
    load_squad_tokens, SquadTestConverter
from trainer import SquadTrainer
from metrics import SquadMetric
from utils import evaluate

parser = ArgumentParser()
parser.add_argument('--epoch', default=100, type=int)
parser.add_argument('--batch', default=64, type=int)
args = parser.parse_args()


spacy_en = spacy.load('en_core_web_sm',
                      disable=['vectors', 'textcat', 'tagger', 'parser', 'ner'])


def tokenizer(x): return [token.text.lower() for token in spacy_en(x) if not token.is_space]


PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'


if not os.path.exists('vocab.pkl'):
    squad_tokens = load_squad_tokens('./data/train-v1.1_filtered.txt', tokenizer)
    token_to_index, index_to_token = Vocabulary.build(
        squad_tokens, 5, 3000, (PAD_TOKEN, UNK_TOKEN), 'vocab.pkl')

else:
    token_to_index, index_to_token = Vocabulary.load('vocab.pkl')

batch_size = args.batch  # Batch size for training.
epochs = args.epoch  # Number of epochs to train for.
latent_dim = 64  # Latent dimensionality of the encoding space.
num_encoder_tokens = len(token_to_index)

print('Number of unique input tokens:', num_encoder_tokens)

model = LightQANet(len(token_to_index), latent_dim, latent_dim).build()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
dataset = SquadReader('./data/train-v1.1_filtered.txt')
converter = SquadConverter(token_to_index, PAD_TOKEN, UNK_TOKEN)
train_generator = Iterator(dataset, batch_size, converter)
trainer = SquadTrainer(model, train_generator, epochs)
trainer.run()
model.save('s2s.h5')

metric = SquadMetric()
dataset = SquadReader('data/dev-v1.1_filtered.txt')
converter = SquadTestConverter(token_to_index, 1, '<pad>', 3)
dev_generator = Iterator(dataset, batch_size, converter, False, False)
em_score, f1_score = evaluate(model, dev_generator, metric, 1, 2, index_to_token)
print('EM: {}, F1: {}'.format(em_score, f1_score))
