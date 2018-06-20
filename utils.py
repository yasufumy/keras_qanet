def char_span_to_token_span(token_offsets, char_start, char_end):
    if char_start < 0:
        return (-1, -1), False

    error = False

    start_index = 0
    while start_index < len(token_offsets) and token_offsets[start_index][0] < char_start:
        start_index += 1
    if token_offsets[start_index][0] > char_start:
        start_index -= 1
    if token_offsets[start_index][0] != char_start:
        error = True

    end_index = start_index
    while end_index < len(token_offsets) and token_offsets[end_index][1] < char_end:
        end_index += 1
    if token_offsets[end_index][1] != char_end:
        error = True
    return (start_index, end_index), error


def get_spans(contexts, starts, ends):
    spans = []
    for context, start, end in zip(contexts, starts, ends):
        context_offsets = [(token.idx, token.idx + len(token.text)) for token in context]
        span, error = char_span_to_token_span(context_offsets, start, end)
        if error:
            ...

        spans.append(span)

    return spans


def tokenizer(x): return [(token.idx, token.text) for token in spacy_en(x) if not token.is_space]


if __name__ == '__main__':
    import os
    import csv
    import pickle
    from allennlp.data.dataset_readers import SquadReader

    import spacy

    spacy_en = spacy.load('en_core_web_sm',
                          disable=['vectors', 'textcat', 'tagger', 'parser', 'ner'])

    if not os.path.exists('data_list.pkl'):
        with open('data/train-v2.0.txt') as f:
            data = [row for row in csv.reader(f, delimiter='\t')]
        data = [[tokenizer(x[0]), int(x[2]), int(x[3]), x[4]]
                for x in data]
        contexts, char_starts, char_ends, answers = zip(*data)
        with open('data_list.pkl', 'wb') as f:
            pickle.dump((contexts, char_starts, char_ends, answers), f)
    else:
        with open('data_list.pkl', 'rb') as f:
            contexts, char_starts, char_ends, answers = pickle.load(f)

    spans = get_spans(contexts, char_starts, char_ends, answers)

    data = SquadReader().read('/Users/smap11/Desktop/train-v2.0.json')

    for span, x in zip(spans, data):
        if 'span_start' in x.fields:
            if span[0] != x.fields['span_start'].sequence_index:
                print('start')
                print(span[0], x.fields['span_start'])
            if span[1] != x.fields['span_end'].sequence_index:
                print('end')
                print(span[1], x.fields['span_end'])
