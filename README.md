# Light QANet for SQuAD

This is a lighter QANet for SQuAD. All models are implemented by Keras.

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


vocab_file= 'vocab.pkl'

PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
min_freq = 5
max_size = 30000

if not os.path.exists(vocab_file):
    spacy_en = spacy.load('en_core_web_sm',
                          disable=['vectors', 'textcat', 'tagger', 'parser', 'ner'])

    def tokenizer(x): return [token.text.lower() for token in spacy_en(x) if not token.is_space]

    train_file = '/path/to/train.txt'
    squad_tokens = load_squad_tokens(train_file)
    token_to_index, index_to_token = Vocabulary.build(
        squad_tokens, min_freq, max_size, (PAD_TOKEN, UNK_TOKEN), vocab_file)
else:
    token_to_index, index_to_token = Vocabulary.load(vocab_file)

```

Training model

```py
from models import LightQANet
from data SquadReader, Iterator, SquadConverter
from trainer SquadTrainer

vocab_size = 30000
hidden_size = embed_size = 128
batch_size = 256

train_file = '/path/to/train.txt'

model = LightQANet(vocab_size, embed_size, hidden_size).build()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
dataset = SquadReader(train_file)
converter = SquadConverter(token_to_index, '<pad>', '<unk>')
train_generator = Iterator(dataset, batch_size, converter)
trainer = SquadTrainer(model, trainer_generator, epochs)
trainer.run()
```

Iterating dataset

```py
from data import SquadReader, Iterator, SquadConverter

train_file = '/path/to/train.tsv'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'

dataset = SquadReader(train_file)
converter = SquadConverter(token_to_index, PAD_TOKEN, UNK_TOKEN)
train_generator = Iterator(dataset, batch_size, converter)
```

Evaluation

```py
from utils import evaluate
from metrics import SquadMetric
from data SquadReader, SquadTestConverter, Iterator


test_file = '/path/to/test.txt'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'

metric = SquadMetric()
dataset = SquadReader(test_file)
converter = SquadTestConverter(token_to_index, PAD_TOKEN, UNK_TOKEN)
test_generator = Iterator(dataset, batch_size, converter, repeat=False, shuffle=False)
em_score, f1_score = evalute(model, test_generator, metric, index_to_token)
```

## Install

```
$ git clone https://github.com/yasufumy/squad_project.git
```
