import os
from argparse import ArgumentParser

import numpy as np
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

from models import QANet
from data import SquadReader, Iterator, SquadConverter, Vocabulary,\
    SquadTestConverter
from trainer import SquadTrainer, BatchLearningRateScheduler, ExponentialMovingAverage
from metrics import SquadMetric
from utils import evaluate, dump_graph

from prepare_vocab import PAD_TOKEN, UNK_TOKEN


def main(args):
    token_to_index, index_to_token = Vocabulary.load(args.vocab_file)

    root, _ = os.path.splitext(args.vocab_file)
    basepath, basename = os.path.split(root)
    embed_path = f'{basepath}/embedding_{basename}.npy'
    embeddings = np.load(embed_path) if os.path.exists(embed_path) else None

    batch_size = args.batch  # Batch size for training.
    epochs = args.epoch  # Number of epochs to train for.

    model = QANet(len(token_to_index), args.embed, args.hidden, args.num_heads,
                  encoder_num_blocks=args.encoder_layer, encoder_num_convs=args.encoder_conv,
                  output_num_blocks=args.output_layer, output_num_convs=args.output_conv,
                  dropout=args.dropout, embeddings=embeddings).build()
    opt = Adam(lr=0.001, beta_1=0.8, beta_2=0.999, epsilon=1e-7, clipnorm=5.)
    model.compile(optimizer=opt,
                  loss=['sparse_categorical_crossentropy',
                        'sparse_categorical_crossentropy', None, None], loss_weights=[1, 1, 0, 0])
    train_dataset = SquadReader(args.train_path)
    dev_dataset = SquadReader(args.dev_path)
    converter = SquadConverter(token_to_index, PAD_TOKEN, UNK_TOKEN, lower=args.lower)
    train_generator = Iterator(train_dataset, batch_size, converter)
    dev_generator = Iterator(dev_dataset, batch_size, converter)
    trainer = SquadTrainer(model, train_generator, epochs, dev_generator,
                           './model/lightqanet.{epoch:02d}-{val_loss:.2f}.h5')
    trainer.add_callback(BatchLearningRateScheduler())
    trainer.add_callback(ExponentialMovingAverage(0.999))
    if args.use_tensorboard:
        trainer.add_callback(TensorBoard(log_dir='./graph', batch_size=batch_size))
    history = trainer.run()
    dump_graph(history, 'loss_graph.png')

    metric = SquadMetric()
    test_dataset = SquadReader(args.test_path)
    converter = SquadTestConverter(token_to_index, PAD_TOKEN, UNK_TOKEN, lower=args.lower)
    test_generator = Iterator(test_dataset, args.batch, converter, False, False)
    em_score, f1_score = evaluate(model, test_generator, metric, index_to_token)
    print('EM: {}, F1: {}'.format(em_score, f1_score))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--batch', default=32, type=int)
    parser.add_argument('--embed', default=300, type=int)
    parser.add_argument('--hidden', default=96, type=int)
    parser.add_argument('--num-heads', default=1, type=int)
    parser.add_argument('--encoder-layer', default=1, type=int)
    parser.add_argument('--encoder-conv', default=4, type=int)
    parser.add_argument('--output-layer', default=7, type=int)
    parser.add_argument('--output-conv', default=2, type=int)
    parser.add_argument('--dropout', default=.1, type=float)
    parser.add_argument('--train-path', default='./data/train-v1.1_filtered_train.txt', type=str)
    parser.add_argument('--dev-path', default='./data/train-v1.1_filtered_dev.txt', type=str)
    parser.add_argument('--test-path', default='./data/dev-v1.1_filtered.txt', type=str)
    parser.add_argument('--vocab-file', default='./data/vocab_question_context_min-freq10_max_size.pkl', type=str)
    parser.add_argument('--lower', default=False, action='store_true')
    parser.add_argument('--use-tensorboard', default=False, action='store_true')
    args = parser.parse_args()
    main(args)
