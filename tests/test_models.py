from unittest import TestCase

import numpy as np

from models import SquadBaseline


class TestSquadBaseline(TestCase):
    def test_build(self):
        model, inference = SquadBaseline(vocab_size=30000,
                                         embed_size=256,
                                         hidden_size=128,
                                         categories=3).build()
        question = np.array([[1, 2, 3, 4, 5, 0, 0, 0],
                             [1, 2, 3, 4, 5, 6, 7, 8]])
        context = np.array([[1, 2, 3, 4, 5, 6],
                            [1, 2, 0, 0, 0, 0]])

        decode_tokens = inference(question, context)

        self.assertEqual(len(decode_tokens), context.shape[1])
        self.assertEqual(len(decode_tokens[0]), context.shape[0])
