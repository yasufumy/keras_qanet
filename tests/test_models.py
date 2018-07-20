from unittest import TestCase

from keras import backend as K
from models import QANet


class TestQANet(TestCase):
    def test_build(self):
        vocab_size = 3000
        embed_size = filters = 96
        context_limit = 40
        query_limit = 5
        num_heads = 1
        model = QANet(
            vocab_size, embed_size, filters, num_heads, context_limit, query_limit,
            dropout=.1, encoder_layer_size=1, encoder_conv_blocks=3,
            output_layer_size=7, output_conv_blocks=2).build()
        query_input, context_input = model.inputs
        start_prob, end_prob, S_bar, S_T = model.outputs

        self.assertTupleEqual(K.int_shape(query_input), (None, query_limit))
        self.assertTupleEqual(K.int_shape(context_input), (None, context_limit))
        self.assertTupleEqual(K.int_shape(start_prob), (None, context_limit))
        self.assertTupleEqual(K.int_shape(end_prob), (None, context_limit))
