import os
from argparse import ArgumentParser

import spacy
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

from models import LightQANet
from data import SquadReader, Iterator, SquadConverter, Vocabulary,\
    load_squad_tokens, SquadTestConverter
from trainer import SquadTrainer
from metrics import SquadMetric
from utils import evaluate

parser = ArgumentParser()
parser.add_argument('--epoch', default=100, type=int)
parser.add_argument('--batch', default=64, type=int)
parser.add_argument('--train-path', default='./data/train-v1.1_filtered_train.txt', type=str)
parser.add_argument('--dev-path', default='./data/train-v1.1_filtered_dev.txt', type=str)
parser.add_argument('--test-path', default='./data/dev-v1.1_filtered.txt', type=str)
parser.add_argument('--min-freq', default=5, type=int)
parser.add_argument('--max-size', default=30000, type=int)
parser.add_argument('--vocab-file', default='vocab.pkl', type=str)
parser.add_argument('--use-tensorboard', default=False, type=bool)
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

model = LightQANet(len(token_to_index), latent_dim, latent_dim).build()
opt = Adam(lr=0.001, beta_1=0.8, beta_2=0.999, epsilon=1e-7, clipnorm=5.)
model.compile(optimizer=opt,
              loss=['sparse_categorical_crossentropy',
                    'sparse_categorical_crossentropy'], loss_weights=[1, 1])
train_dataset = SquadReader(args.train_path)
dev_dataset = SquadReader(args.dev_path)
converter = SquadConverter(token_to_index, PAD_TOKEN, UNK_TOKEN)
train_generator = Iterator(train_dataset, batch_size, converter)
dev_generator = Iterator(dev_dataset, batch_size, converter)
trainer = SquadTrainer(model, train_generator, epochs, dev_generator,
                       'lightqanet.h5')
if args.use_tensorboard:
    trainer.add_callback(TensorBoard(log_dir='./graph', batch_size=batch_size))
trainer.run()

metric = SquadMetric()
test_dataset = SquadReader(args.test_path)
converter = SquadTestConverter(token_to_index, PAD_TOKEN, UNK_TOKEN)
test_generator = Iterator(test_dataset, args.batch, converter, False, False)
em_score, f1_score = evaluate(model, test_generator, metric, index_to_token)
print('EM: {}, F1: {}'.format(em_score, f1_score))
