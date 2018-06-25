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
from utils import make_vocab

token_to_index, index_to_token = make_vocab('/path/to/train.tsv')
```

Building model

```py
from models import SquadBaseline

vocab_size = len(token_to_index)
hidden_size = embed_size = 128
batch_size = 256

model, inference = SquadBaseline(vocab_size, embed_size, hidden_size)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
```

Iterating dataset

```py
from data import SquadReader, SquadIterator, SquadConverter

dataset = SquadReader('/path/to/train.tsv')
converter = SquadConverter(token_to_index, 1, '<pad>', 3)
train_generator = SquadIterator(dataset, batch_size, converter)
```

Training

```py
from trainer import SquadTrainer

trainer = SquadTrainer(model, trainer_generator, epochs)
trainer.run()
```

Evaluation

```py
from evaluator import SquadEvaluator
from metrics import SquadMetric

metric = SquadMetric()
evaluator = SquadEvaluator(inference, dev_generator, metric)
```

## Install

```
$ git clone https://github.com/yasufumy/squad_project.git
```
