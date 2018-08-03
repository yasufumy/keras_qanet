import os
from argparse import ArgumentParser

import tensorflow as tf
from keras import backend as K
import numpy as np
# from keras.optimizers import Adam
from keras.callbacks import TensorBoard

from models import QANet
from data import SquadReader, Iterator, SquadConverter, Vocabulary
from trainer import SquadTrainer  # , BatchLearningRateScheduler  # , ExponentialMovingAverage
from utils import dump_graph

from prepare_vocab import PAD_TOKEN, UNK_TOKEN


def tensorflow_ops(model, train_generator, dev_generator):
    sess = tf.Session()
    K.set_session(sess)

    input_tensors = model.input
    output_tensors = model.output
    start_labels = tf.placeholder(tf.int32, shape=(None,))
    end_labels = tf.placeholder(tf.int32, shape=(None,))

    def position_loss(y, t):
        return tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=t, logits=y))

    loss = position_loss(output_tensors[0], start_labels) + position_loss(output_tensors[1], end_labels)
    for l2_loss in model.losses:
        loss += l2_loss

    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.minimum(0.001, 0.001 / tf.log(tf.cast(1000 - 1, tf.float32)
                                                     * tf.log(tf.cast(global_step, tf.float32) + 1)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.8, beta2=0.999, epsilon=1e-7)

    grads, tvars = zip(*optimizer.compute_gradients(loss, colocate_gradients_with_ops=True))
    clipped_grads, _ = tf.clip_by_global_norm(grads, 5.)
    apply_gradient_op = optimizer.apply_gradients(zip(clipped_grads, tvars), global_step=global_step)

    ema = tf.train.ExponentialMovingAverage(0.9999, global_step)
    with tf.control_dependencies([apply_gradient_op]):
        training_op = ema.apply(tf.trainable_variables())

    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    with sess.as_default():
        for i in range(70):
            inputs, outputs = next(train_generator)
            loss_val, _, global_step_val = sess.run([loss, training_op, global_step], feed_dict={
                input_tensors[0]: inputs[0], input_tensors[1]: inputs[1],
                start_labels: outputs[0], end_labels: outputs[1]})
            print(f'{i}: {loss_val}')


def main(args):
    token_to_index, index_to_token = Vocabulary.load(args.vocab_file)

    root, _ = os.path.splitext(args.vocab_file)
    basepath, basename = os.path.split(root)
    embed_path = f'{basepath}/embedding_{basename}.npy'
    embeddings = np.load(embed_path) if os.path.exists(embed_path) else None

    batch_size = args.batch  # Batch size for training.
    epochs = args.epoch  # Number of epochs to train for.

    model = QANet(len(token_to_index), args.embed, args.hidden, args.num_heads,
                  encoder_num_blocks=args.encoder_layer, encoder_num_convs=args.encoder_conv,
                  output_num_blocks=args.output_layer, output_num_convs=args.output_conv,
                  dropout=args.dropout, embeddings=embeddings).build()
    # opt = Adam(lr=0.0001)
    # opt = Adam(lr=0.001, beta_1=0.8, beta_2=0.999, epsilon=1e-7)
    # model.compile(optimizer=opt,
    #               loss=['sparse_categorical_crossentropy',
    #                     'sparse_categorical_crossentropy', None, None], loss_weights=[1, 1, 0, 0])
    train_dataset = SquadReader(args.train_path)
    dev_dataset = SquadReader(args.dev_path)
    converter = SquadConverter(token_to_index, PAD_TOKEN, UNK_TOKEN, lower=args.lower)
    train_generator = Iterator(train_dataset, batch_size, converter)
    dev_generator = Iterator(dev_dataset, batch_size, converter)
    tensorflow_ops(model, train_generator, dev_generator)
    model.save_weights('./model/qanet.h5')
    return
    trainer = SquadTrainer(model, train_generator, epochs, dev_generator,
                           './model/qanet.{epoch:02d}-{val_loss:.2f}.h5')
    # trainer.add_callback(BatchLearningRateScheduler())
    # trainer.add_callback(ExponentialMovingAverage(0.999))
    if args.use_tensorboard:
        trainer.add_callback(TensorBoard(log_dir='./graph', batch_size=batch_size))
    history = trainer.run()
    dump_graph(history, 'loss_graph.png')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--batch', default=32, type=int)
    parser.add_argument('--embed', default=300, type=int)
    parser.add_argument('--hidden', default=96, type=int)
    parser.add_argument('--num-heads', default=1, type=int)
    parser.add_argument('--encoder-layer', default=1, type=int)
    parser.add_argument('--encoder-conv', default=4, type=int)
    parser.add_argument('--output-layer', default=7, type=int)
    parser.add_argument('--output-conv', default=2, type=int)
    parser.add_argument('--dropout', default=.1, type=float)
    parser.add_argument('--train-path', default='./data/train-v1.1_filtered_train.txt', type=str)
    parser.add_argument('--dev-path', default='./data/train-v1.1_filtered_dev.txt', type=str)
    parser.add_argument('--test-path', default='./data/dev-v1.1_filtered.txt', type=str)
    parser.add_argument('--vocab-file', default='./data/vocab_question_context_min-freq10_max_size.pkl', type=str)
    parser.add_argument('--lower', default=False, action='store_true')
    parser.add_argument('--use-tensorboard', default=False, action='store_true')
    args = parser.parse_args()
    main(args)
