from unittest import TestCase

import tensorflow as tf
import numpy as np

from layers import PositionEmbedding


class TestPositionEmbedding(TestCase):
    def setUp(self):
        self.pe = PositionEmbedding()

    def test_call(self):
        inputs = np.random.randn(64, 400, 128).astype(np.float32)
        outputs = self.pe(tf.Variable(inputs))
        batch_size, seq_len, hidden_size = outputs.shape
        self.assertEqual(batch_size, 64)
        self.assertEqual(seq_len, 400)
        self.assertEqual(hidden_size, 128)
