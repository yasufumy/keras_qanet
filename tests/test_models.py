from unittest import TestCase

from keras import backend as K
from models import LightQANet


class TestLightQANet(TestCase):
    def test_build(self):
        vocab_size = 3000
        embed_size = filters = 32
        context_limit = 40
        query_limit = 5
        model = LightQANet(
            vocab_size, embed_size, filters, context_limit, query_limit).build()
        query_input, context_input = model.inputs
        start_prob, end_prob = model.outputs

        self.assertTupleEqual(K.int_shape(query_input), (None, query_limit))
        self.assertTupleEqual(K.int_shape(context_input), (None, context_limit))
        self.assertTupleEqual(K.int_shape(start_prob), (None, context_limit))
        self.assertTupleEqual(K.int_shape(end_prob), (None, context_limit))
