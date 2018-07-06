import os
from argparse import ArgumentParser

import spacy
from keras.optimizers import Adam

from models import LightQANet
from data import SquadReader, Iterator, SquadConverter, Vocabulary,\
    load_squad_tokens, SquadTestConverter
from trainer import SquadTrainer
from metrics import SquadMetric
from utils import evaluate

parser = ArgumentParser()
parser.add_argument('--epoch', default=100, type=int)
parser.add_argument('--batch', default=64, type=int)
parser.add_argument('--dev-batch', default=64, type=int)
parser.add_argument('--train-path', default='./data/train-v1.1_filtered.txt', type=str)
parser.add_argument('--dev-path', default='./data/dev-v1.1_filtered.txt', type=str)
parser.add_argument('--min-freq', default=5, type=int)
parser.add_argument('--max-size', default=30000, type=int)
parser.add_argument('--vocab-file', default='vocab.pkl', type=str)
args = parser.parse_args()


PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'

if not os.path.exists(args.vocab_file):
    spacy_en = spacy.load('en_core_web_sm',
                          disable=['vectors', 'textcat', 'tagger', 'parser', 'ner'])

    def tokenizer(x): return [token.text.lower() for token in spacy_en(x) if not token.is_space]

    squad_tokens = load_squad_tokens(args.train_path, tokenizer)
    token_to_index, index_to_token = Vocabulary.build(
        squad_tokens, args.min_freq, args.max_size, (PAD_TOKEN, UNK_TOKEN),
        args.vocab_file)

else:
    token_to_index, index_to_token = Vocabulary.load(args.vocab_file)

batch_size = args.batch  # Batch size for training.
epochs = args.epoch  # Number of epochs to train for.
latent_dim = 64  # Latent dimensionality of the encoding space.
num_encoder_tokens = len(token_to_index)

print('Number of unique input tokens:', num_encoder_tokens)

model = LightQANet(len(token_to_index), latent_dim, latent_dim).build()
opt = Adam(lr=0.001, beta_1=0.8, beta_2=0.999, epsilon=1e-7, clipnorm=5.)
model.compile(optimizer=opt,
              loss=['sparse_categorical_crossentropy',
                    'sparse_categorical_crossentropy'], loss_weights=[1, 1])
dataset = SquadReader(args.train_path)
converter = SquadConverter(token_to_index, PAD_TOKEN, UNK_TOKEN)
train_generator = Iterator(dataset, batch_size, converter)
trainer = SquadTrainer(model, train_generator, epochs)
trainer.run()
model.save('s2s.h5')

metric = SquadMetric()
dataset = SquadReader(args.dev_path)
converter = SquadTestConverter(token_to_index, PAD_TOKEN, UNK_TOKEN)
dev_generator = Iterator(dataset, args.dev_batch, converter, False, False)
em_score, f1_score = evaluate(model, dev_generator, metric, index_to_token)
print('EM: {}, F1: {}'.format(em_score, f1_score))
