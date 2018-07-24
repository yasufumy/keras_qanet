import os
import pickle
from argparse import ArgumentParser

import numpy as np
import spacy
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

from models import DependencyNet
from data import SquadReader, Iterator, SquadDepConverter, Vocabulary,\
    load_squad_tokens
from trainer import SquadTrainer, BatchLearningRateScheduler, ExponentialMovingAverage
from metrics import SquadMetric
from utils import evaluate, dump_graph, extract_embeddings

parser = ArgumentParser()
parser.add_argument('--epoch', default=100, type=int)
parser.add_argument('--batch', default=32, type=int)
parser.add_argument('--embed', default=300, type=int)
parser.add_argument('--hidden', default=96, type=int)
parser.add_argument('--num-heads', default=1, type=int)
parser.add_argument('--encoder-layer', default=1, type=int)
parser.add_argument('--encoder-conv', default=4, type=int)
parser.add_argument('--dropout', default=.1, type=float)
parser.add_argument('--train-path', default='./data/train-v1.1_filtered_train.txt', type=str)
parser.add_argument('--dev-path', default='./data/train-v1.1_filtered_dev.txt', type=str)
parser.add_argument('--test-path', default='./data/dev-v1.1_filtered.txt', type=str)
parser.add_argument('--embed-file', default='./data/squad_embedding.npy', type=str)
parser.add_argument('--embed-array-path', default='./data/wiki.en.vec.npy', type=str)
parser.add_argument('--embed-dict-path', default='./data/wiki.en.vec.dict', type=str)
parser.add_argument('--min-freq', default=10, type=int)
parser.add_argument('--max-size', default=None, type=int)
parser.add_argument('--vocab-file', default='vocab.pkl', type=str)
parser.add_argument('--use-tensorboard', default=False, action='store_true')
args = parser.parse_args()


PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'

if not os.path.exists(args.vocab_file):
    spacy_en = spacy.load('en_core_web_sm',
                          disable=['vectors', 'textcat', 'tagger', 'parser', 'ner'])

    def tokenizer(x): return [token.text.lower() for token in spacy_en(x) if not token.is_space]

    squad_tokens = load_squad_tokens(args.train_path, tokenizer, indices=[1])
    token_to_index, index_to_token = Vocabulary.build(
        squad_tokens, args.min_freq, args.max_size, (PAD_TOKEN, UNK_TOKEN),
        args.vocab_file)

else:
    token_to_index, index_to_token = Vocabulary.load(args.vocab_file)

if not os.path.exists(args.embed_file):
    if os.path.exists(args.embed_dict_path):
        with open(args.embed_dict_path, 'rb') as f:
            big_token_to_index = pickle.load(f)
        embeddings = extract_embeddings(token_to_index, big_token_to_index, np.load(args.embed_array_path))
        np.save(args.embed_file, embeddings)
    else:
        embeddings = None
else:
    embeddings = np.load(args.embed_file)

batch_size = args.batch  # Batch size for training.
epochs = args.epoch  # Number of epochs to train for.
converter = SquadDepConverter(token_to_index, PAD_TOKEN, UNK_TOKEN)

model = DependencyNet(len(token_to_index), args.embed, len(converter._dep_to_index),
                      args.hidden, args.num_heads,
                      dropout=args.dropout, num_blocks=args.encoder_layer,
                      num_convs=args.encoder_conv, embeddings=embeddings).build()
opt = Adam(lr=0.001, beta_1=0.8, beta_2=0.999, epsilon=1e-7, clipnorm=5.)
model.compile(optimizer=opt, loss=['sparse_categorical_crossentropy'],
              metrics=['sparse_categorical_accuracy'])
train_dataset = SquadReader(args.train_path)
dev_dataset = SquadReader(args.dev_path)
train_generator = Iterator(train_dataset, batch_size, converter)
dev_generator = Iterator(dev_dataset, batch_size, converter)
trainer = SquadTrainer(model, train_generator, epochs, dev_generator,
                       './model/dep.{epoch:02d}-{val_loss:.2f}.h5')
trainer.add_callback(BatchLearningRateScheduler())
trainer.add_callback(ExponentialMovingAverage(0.999))
if args.use_tensorboard:
    trainer.add_callback(TensorBoard(log_dir='./graph', batch_size=batch_size))
history = trainer.run()
dump_graph(history, 'loss_graph.png')

metric = SquadMetric()
test_dataset = SquadReader(args.test_path)
test_generator = Iterator(test_dataset, args.batch, converter, False, False)
em_score, f1_score = evaluate(model, test_generator, metric, index_to_token)
print('EM: {}, F1: {}'.format(em_score, f1_score))
