# Baselines for SQuAD

This is baseline models for SQuAD. All models are implemented by Keras.

## Description

Easy to start training SQuAD. Building vocabulary, loading dataset,
Visualizing attention scores

## Requirement

- Python 3.6+
- TensorFlow, Keras, NumPy, spaCy

## Usage

Buidling vocabulary

```py
import os
from data import Vocabulary, load_squad_tokens
import spacy

spacy_en = spacy.load('en_core_web_sm',
                      disable=['vectors', 'textcat', 'tagger', 'parser', 'ner'])

def tokenizer(x): return [token.text.lower() for token in spacy_en(x) if not token.is_space]

if not os.path.exists('vocab.pkl'):
    squad_tokens = load_squad_tokens('/path/to/train.txt')
    token_to_index, index_to_token = Vocabulary.build(
        squad_tokens, 5, 30000, ('<pad>', '<unk>'), 'vocab.pkl')
else:
    token_to_index, index_to_token = Vocabulary.load('vocab.pkl')

```

Training model

```py
from models import SquadBaseline
from data SquadReader, Iterator, SquadConverter
from trainer SquadTrainer

vocab_size = len(token_to_index)
hidden_size = embed_size = 128
batch_size = 256

model, inference = SquadBaseline(vocab_size, embed_size, hidden_size).build()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
dataset = SquadReader('/path/to/train.txt')
converter = SquadConverter(token_to_index, 1, '<pad>', 3)
train_generator = Iterator(dataset, batch_size, converter)
trainer = SquadTrainer(model, trainer_generator, epochs)
trainer.run()
```

Iterating dataset

```py
from data import SquadReader, SquadIterator, SquadConverter

dataset = SquadReader('/path/to/train.tsv')
converter = SquadConverter(token_to_index, 1, '<pad>', 3)
train_generator = SquadIterator(dataset, batch_size, converter)
```

Evaluation

```py
from utils import evaluate
from metrics import SquadMetric
from data SquadReader, SquadTestConverter, Iterator

metric = SquadMetric()
dataset = SquadReader('/path/to/test.txt')
converter = SquadTestConverter(token_to_index, 1, '<pad>', 3)
test_generator = Iterator(dataset, batch_size, converter, False, False)
em_score, f1_score = evalute(inference, test_generator, metric, 1, 2, index_to_token)
```

## Install

```
$ git clone https://github.com/yasufumy/squad_project.git
```
