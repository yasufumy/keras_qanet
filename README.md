# Baselines for SQuAD


## Usage

```py
from models import SquadBaseline
from preprocessing import make_vocab
from trainer import Trainer
from evaluator import Evaluator
from utils import load_tokens, SquadTrainSequence, SquadConverter, SquadMetric

context_and_question_tokens = load_tokens(fields={'context', 'question'})
token_to_index, index_to_token = make_vocab(context_and_question_tokens)
converter = SquadConverter(token_to_index)

epochs = 10
vocab_size = len(token_to_index)
hidden_size = embed_size = 128
batch_size = 256


model, inference = SquadBaseline(vocab_size, embed_size, hidden_size)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

train_generator = SquadTrainSequence(
    'path_to_train.tsv', batch_size=batch_size, converter=converter)

Trainer(model, train_generator, epochs).run()

dev_generator = SquadDevSequence(
    'path_to_dev.tsv', batch_size=batch_size, converter=converter)

metric, attention_score = Evaluator(inference, dev_generator, SquadMetric()).run()
```
