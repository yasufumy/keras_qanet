from unittest import TestCase
from utils import char_span_to_token_span, get_spans


class TestUitls(TestCase):
    def test_char_span_to_token_span(self):
        token_offsets = [(0, 4), (5, 6), (7, 11), (12, 14), (15, 16), (17, 21),
                         (21, 22), (23, 26), (27, 31), (32, 37), (38, 47), (47, 48)]
        char_start = 0
        char_end = 11
        span, error = char_span_to_token_span(token_offsets, char_start, char_end)

        self.assertEqual(error, False)
        self.assertEqual(span, (0, 2))

    def test_get_spans(self):
        import spacy
        spacy_en = spacy.load(
            'en_core_web_sm', disable=['vectors', 'textcat', 'tagger', 'parser', 'ner'])
        text = 'Rock n Roll is a risk. You risk being ridiculed.'
        contexts = [[token for token in spacy_en(text) if not token.is_space]]
        spans = get_spans(contexts, [0], [11])

        self.assertEqual(spans[0], (0, 2))
