import os
from argparse import ArgumentParser

import numpy as np

from models import QANet
from data import SquadReader, Iterator, Vocabulary, SquadTestConverter
from metrics import SquadMetric
from utils import evaluate

from prepare_vocab import PAD_TOKEN, UNK_TOKEN


def main(args):
    token_to_index, index_to_token = Vocabulary.load(args.vocab_file)

    root, _ = os.path.splitext(args.vocab_file)
    basepath, basename = os.path.split(root)
    embed_path = f'{basepath}/embedding_{basename}.npy'
    embeddings = np.load(embed_path) if os.path.exists(embed_path) else None

    model = QANet(len(token_to_index), args.embed, args.hidden, args.num_heads,
                  encoder_num_blocks=args.encoder_layer, encoder_num_convs=args.encoder_conv,
                  output_num_blocks=args.output_layer, output_num_convs=args.output_conv,
                  dropout=args.dropout, embeddings=embeddings).build()
    model.load_weights(args.model_path)

    metric = SquadMetric()
    test_dataset = SquadReader(args.test_path)
    converter = SquadTestConverter(token_to_index, PAD_TOKEN, UNK_TOKEN, lower=args.lower)
    test_generator = Iterator(test_dataset, args.batch, converter, False, False)
    em_score, f1_score = evaluate(model, test_generator, metric, index_to_token)
    print('EM: {}, F1: {}'.format(em_score, f1_score))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--batch', default=32, type=int)
    parser.add_argument('--embed', default=300, type=int)
    parser.add_argument('--hidden', default=96, type=int)
    parser.add_argument('--num-heads', default=1, type=int)
    parser.add_argument('--encoder-layer', default=1, type=int)
    parser.add_argument('--encoder-conv', default=4, type=int)
    parser.add_argument('--output-layer', default=7, type=int)
    parser.add_argument('--output-conv', default=2, type=int)
    parser.add_argument('--dropout', default=.1, type=float)
    parser.add_argument('--test-path', default='./data/dev-v1.1_filtered.txt', type=str)
    parser.add_argument('--vocab-file', default='./data/vocab_question_context_min-freq10_max_size.pkl', type=str)
    parser.add_argument('--lower', default=False, action='store_true')
    parser.add_argument('--model-path', type=str)
    args = parser.parse_args()
    main(args)
