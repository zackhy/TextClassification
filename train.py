import os
import time
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import data_helper
import random
from rnn_classifier import rnn_clf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class config(object):
    num_classes = 3
    batch_size = 30
    vocab_size = 20000
    embedding_size = 128
    hidden_size = 128
    num_layers = 3
    learning_rate = 0.0001
    keep_prob = 0.3
    num_epochs = 50
    l2_reg_lambda = 0.005


def run_epoch(data, model, sess, train_op=None):
    input_x, input_y, length = data
    fetches = {'final_state': model.final_state,
               'cost': model.cost,
               'accuracy': model.accuracy}

    if train_op is not None:
        fetches['train_op'] = train_op

    feed_dict = {model.input_x: input_x,
                 model.input_y: input_y,
                 model.sequence_length: length}

    vars = sess.run(fetches, feed_dict)
    cost = vars['cost']
    accuracy = vars['accuracy']

    return cost, accuracy

def main():
    data, labels, idx_2_w, vocab_size = data_helper.load_data(file_path='sample.csv',
                                                              sw_path='stop_words_ch.txt',
                                                              save_path='data/',
                                                              vocab_size=config.vocab_size)

    # Cross validation
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)

    with tf.Graph().as_default():
        with tf.name_scope('Train'):
            with tf.variable_scope('Model', reuse=None):
                m = rnn_clf(num_classes=config.num_classes,
                            batch_size=config.batch_size,
                            vocab_size=vocab_size,
                            embedding_size=config.embedding_size,
                            hidden_size=config.hidden_size,
                            num_layers=config.num_layers,
                            learning_rate=config.learning_rate,
                            keep_prob=config.keep_prob,
                            l2_reg_lambda=config.l2_reg_lambda,
                            is_training=True)
            tf.summary.scalar('Training loss', m.cost)
            tf.summary.scalar('Training accuracy', m.accuracy)

        with tf.name_scope('Valid'):
            with tf.variable_scope('Model', reuse=True):
                mvalid = rnn_clf(num_classes=config.num_classes,
                                 batch_size=config.batch_size,
                                 vocab_size=vocab_size,
                                 embedding_size=config.embedding_size,
                                 hidden_size=config.hidden_size,
                                 num_layers=config.num_layers,
                                 learning_rate=config.learning_rate,
                                 keep_prob=config.keep_prob,
                                 l2_reg_lambda=config.l2_reg_lambda,
                                 is_training=False)
            tf.summary.scalar('Validation loss', mvalid.cost)
            tf.summary.scalar('Validation accuracy', mvalid.accuracy)

        # merged_summary_op = tf.summary.merge_all()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(config.num_epochs):
                start = time.time()
                train_step = 0
                valid_step = 0
                total_train_cost = 0
                total_train_accuracy = 0
                total_valid_cost = 0
                total_valid_accuracy = 0

                train_data = data_helper.batch_iter(x_train, y_train, config.batch_size)
                valid_data = data_helper.batch_iter(x_test, y_test, config.batch_size)

                # Train
                for train_input in train_data:
                    train_cost, train_accuracy = run_epoch(train_input,
                                                           model=m,
                                                           sess=sess,
                                                           train_op=m.train_op)
                    train_step += 1
                    total_train_cost += train_cost
                    total_train_accuracy += train_accuracy

                    if train_step % 200 == 0:
                        print('Batch: {}, Step: {}, Loss: {}, Accuracy: {}'.format(i,
                                                                                   train_step,
                                                                                   train_cost,
                                                                                   train_accuracy))


                # Validation
                for valid_input in valid_data:
                    valid_cost, valid_accuracy = run_epoch(valid_input,
                                                           model=mvalid,
                                                           sess=sess)
                    valid_step += 1
                    total_valid_cost += valid_cost
                    total_valid_accuracy += valid_accuracy

                end = time.time()
                runtime = end - start

                print('Batch: {}, Train loss: {}, Train accuracy: {}'.format(i,
                                                                             total_train_cost / train_step,
                                                                             total_train_accuracy / train_step))
                print('Batch: {}, Valid loss: {}, Valid accuracy: {}'.format(i,
                                                                             total_valid_cost / valid_step,
                                                                             total_valid_accuracy / valid_step))
                print('Run time: {}'.format(runtime))

if __name__ == '__main__':
    main()
