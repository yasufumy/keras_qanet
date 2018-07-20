from unittest import TestCase

import tensorflow as tf
import numpy as np

from layers import PositionEmbedding, MultiHeadAttention, ContextQueryAttention,\
    LayerDropout


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


class TestMultiHeadAttention(TestCase):
    def setUp(self):
        self.attn = MultiHeadAttention(128, 4)

    def test_call(self):
        key_and_value = tf.Variable(np.random.randn(64, 400, 256).astype(np.float32))
        query = tf.Variable(np.random.randn(64, 400, 128).astype(np.float32))
        seq_len = tf.Variable(np.ones((64, 1)).astype(np.int32))
        outputs = self.attn([key_and_value, query, seq_len])
        batch_size, seq_len, hidden_size = outputs.shape
        self.assertEqual(batch_size, 64)
        self.assertEqual(seq_len, 400)
        self.assertEqual(hidden_size, 128)


class TestContextQueryAttention(TestCase):
    def setUp(self):
        self.attn = ContextQueryAttention(128, 400, 50)

    def test_call(self):
        context = tf.Variable(np.random.randn(64, 400, 128).astype(np.float32))
        query = tf.Variable(np.random.randn(64, 50, 128).astype(np.float32))
        seq_len = tf.Variable(np.ones((64, 1)).astype(np.int32))
        output, S_bar, S_T = self.attn([context, query, seq_len, seq_len])
        batch_size, seq_len, hidden_size = output.shape
        self.assertEqual(batch_size, 64)
        self.assertEqual(seq_len, 400)
        self.assertEqual(hidden_size, 128 * 4)


class TestLayerDropout(TestCase):
    def setUp(self):
        self.ratio = 0.2
        self.layer_dropout = LayerDropout(dropout=self.ratio)

    def test_call(self):
        x = tf.Variable(np.random.randn(64, 400, 128).astype(np.float32))
        residual = tf.Variable(np.random.randn(64, 400, 128).astype(np.float32))
        output = self.layer_dropout([x, residual])
        batch_size, seq_len, hidden_size = output.shape
        self.assertEqual(batch_size, 64)
        self.assertEqual(seq_len, 400)
        self.assertEqual(hidden_size, 128)
