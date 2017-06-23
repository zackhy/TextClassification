# -*- coding: utf-8 -*-
import os
import time
import tensorflow as tf
from sklearn.model_selection import train_test_split

import data_helper
from rnn_classifier import rnn_clf
from cnn_classifier import cnn_clf

# Show warnings and errors only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Hyperparameters of RNN classifier
class rnn_config(object):
    num_classes = 3
    batch_size = 1
    vocab_size = 20000
    embedding_size = 10
    hidden_size = 10
    num_layers = 1
    keep_prob = 0.5
    learning_rate = 1e-3
    num_epochs = 5
    l2_reg_lambda = 0.0


# Hyperparameters of CNN classifier
class cnn_config(object):
    sequence_length = 0
    num_classes = 3
    batch_size = 1
    vocab_size = 20000
    embedding_size = 128
    filter_sizes = [3, 4, 5]
    num_filters = 128
    keep_prob = 0.5
    learning_rate = 1e-3
    num_epochs = 50
    l2_reg_lambda = 0.0


def run_epoch(data, model, sess, clf='rnn', train_op=None):
    """
    Run one epoch
    :param data: mini-batch data
    :param model: the model for running
    :param sess: the tensorflow session
    :param clf: the type of neural network classifier
    :param train_op: the optimizer for training.
    :return: total loss, accuracy and L2 loss
    """
    if clf == 'rnn':
        input_x, input_y, length = data
    elif clf == 'cnn':
        input_x, input_y = data

    fetches = {'cost': model.cost,
               'accuracy': model.accuracy,
               'l2_loss': model.l2_loss}
    feed_dict = {model.input_x: input_x,
                 model.input_y: input_y}

    if clf == 'rnn':
        fetches['final_state'] = model.final_state
        feed_dict[model.sequence_length] = length

    if train_op is not None:
        fetches['train_op'] = train_op

    vars = sess.run(fetches, feed_dict)
    cost = vars['cost']
    accuracy = vars['accuracy']
    l2_loss = vars['l2_loss']

    return cost, accuracy, l2_loss

def train(clf='rnn'):
    """
    Train and validate
    :param clf: the type of neural network classifier
    :return: nothing
    """
    if clf == 'rnn':
        config = rnn_config
    elif clf == 'cnn':
        config = cnn_config
    else:
        raise ValueError("clf should be either 'rnn' or 'cnn'")

    data, labels, idx_2_w, vocab_size, max_length = data_helper.load_data(file_path='test.csv',
                                                                          sw_path='stop_words_ch.txt',
                                                                          save_path='data/',
                                                                          vocab_size=config.vocab_size)

    config.vocab_size = min(vocab_size, config.vocab_size)
    if clf == 'cnn':
        config.sequence_length = max_length


    # Cross validation
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)

    with tf.Graph().as_default():
        with tf.name_scope('Train'):
            with tf.variable_scope('Model', reuse=None):
                if clf == 'rnn':
                    m = rnn_clf(config, is_training=True)
                elif clf == 'cnn':
                    m = cnn_clf(config, is_training=True)

            tf.summary.scalar('Training loss', m.cost)
            tf.summary.scalar('Training accuracy', m.accuracy)

        with tf.name_scope('Valid'):
            with tf.variable_scope('Model', reuse=True):
                if clf == 'rnn':
                    mvalid = rnn_clf(config, is_training=False)
                elif clf == 'cnn':
                    mvalid = cnn_clf(config, is_training=False)

            tf.summary.scalar('Validation loss', mvalid.cost)
            tf.summary.scalar('Validation accuracy', mvalid.accuracy)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            print('Start training....')
            for i in range(rnn_config.num_epochs):
                start = time.time()
                train_step = 0
                valid_step = 0
                total_train_cost = 0
                total_train_accuracy = 0
                total_valid_cost = 0
                total_valid_accuracy = 0

                train_data = data_helper.batch_iter(x_train, y_train, config.batch_size, max_length, clf=clf)
                valid_data = data_helper.batch_iter(x_test, y_test, config.batch_size, max_length, clf=clf)

                # Train
                for train_input in train_data:
                    train_cost, train_accuracy, _ = run_epoch(train_input,
                                                              model=m,
                                                              sess=sess,
                                                              train_op=m.train_op,
                                                              clf=clf)
                    train_step += 1
                    total_train_cost += train_cost
                    total_train_accuracy += train_accuracy

                    if train_step % 10 == 0:
                        print('Epoch: {}, Step: {}, Loss: {}, Accuracy: {}'.format(i,
                                                                                   train_step,
                                                                                   train_cost,
                                                                                   train_accuracy))


                # Validation
                for valid_input in valid_data:
                    valid_cost, valid_accuracy, _ = run_epoch(valid_input,
                                                              model=mvalid,
                                                              sess=sess,
                                                              clf=clf)
                    valid_step += 1
                    total_valid_cost += valid_cost
                    total_valid_accuracy += valid_accuracy

                end = time.time()
                runtime = end - start

                print('Epoch: {}, Train loss: {}, Train accuracy: {}'.format(i,
                                                                             total_train_cost / train_step,
                                                                             total_train_accuracy / train_step))
                print('Epoch: {}, Valid loss: {}, Valid accuracy: {}'.format(i,
                                                                             total_valid_cost / valid_step,
                                                                             total_valid_accuracy / valid_step))
                print('Run time: {}'.format(runtime))

if __name__ == '__main__':
    train(clf='cnn')
