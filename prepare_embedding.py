import os
import pickle
from argparse import ArgumentParser

import numpy as np

from data import Vocabulary
from utils import extract_embeddings, save_word_embedding_as_npy


def main(args):
    token_to_index, _ = Vocabulary.load(args.vocab_path)

    if os.path.exists(args.embed_array_path) and os.path.exists(args.embed_dict_path):
        with open(args.embed_dict_path, 'rb') as f:
            pretrained_token_to_index = pickle.load(f)
        embeddings = extract_embeddings(token_to_index, pretrained_token_to_index,
                                        np.load(args.embed_array_path))
    else:
        if os.path.exists(args.embed_path):
            pretrained_token_to_index, embeddings = save_word_embedding_as_npy(args.embed_path, args.dim)
        else:
            raise FileNotFoundError('Please download pre-trained embedding file')
    root, _ = os.path.splitext(args.vocab_path)
    basepath, basename = os.path.split(root)
    filename = f'{basepath}/embedding_{basename}.npy'
    np.save(filename, embeddings)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--vocab-path', default='./data/vocab_question_context_min-freq10_max_size.pkl', type=str)
    parser.add_argument('--embed-path', default='./data/wiki.en.vec', type=str)
    parser.add_argument('--dim', default=300, type=int)
    parser.add_argument('--embed-array-path', default='./data/wiki.en.vec.npy', type=str)
    parser.add_argument('--embed-dict-path', default='./data/wiki.en.vec.dict', type=str)
    args = parser.parse_args()

    main(args)
