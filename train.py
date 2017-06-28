# -*- coding: utf-8 -*-
import os
import csv
import time
import json
import pickle as pkl
import tensorflow as tf
from tensorflow.contrib import learn

import data_helper
from rnn_classifier import rnn_clf
from cnn_classifier import cnn_clf

try:
    from sklearn.model_selection import train_test_split
except ImportError as e:
    error = "Please install scikit-learn."
    print(str(e) + ': ' + error)
    sys.exit()

# Show warnings and errors only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Hyperparameters of RNN classifier
class rnn_config(object):
    num_classes = 3
    batch_size = 32
    min_frequency = 0
    embedding_size = 1500
    hidden_size = 1500
    num_layers = 3
    keep_prob = 0.5
    learning_rate = 1e-3
    num_epochs = 5
    l2_reg_lambda = 0.01


# Hyperparameters of CNN classifier
class cnn_config(object):
    num_classes = 3
    batch_size = 32
    min_frequency = 0
    embedding_size = 128
    filter_sizes = [3, 4, 5]
    num_filters = 256
    keep_prob = 0.4
    learning_rate = 1e-3
    num_epochs = 51
    l2_reg_lambda = 0.01

def train(clf='rnn', outdir='result'):
    """ Train the classifier and implement cross-validation """
    if clf == 'rnn':
        config = rnn_config
    elif clf == 'cnn':
        config = cnn_config
    else:
        raise ValueError("clf should be either 'rnn' or 'cnn'")

    # ----------------------------------- Load data -----------------------------------
    data, labels, vocab_processor = data_helper.load_data(file_path='test.csv',
                                                          sw_path='stop_words_ch.txt',
                                                          min_frequency=config.min_frequency,
                                                          language='ch')

    # Save vocabulary processor
    vocab_processor.save(os.path.join(outdir, 'vocab'))

    config.vocab_size = len(vocab_processor.vocabulary_._mapping)
    max_length = vocab_processor.max_document_length

    if clf == 'cnn':
        config.max_length = max_length

    # Cross validation
    x_train, x_valid, y_train, y_valid = train_test_split(data, labels, test_size=0.1, random_state=42)

    # ----------------------------------- Training -----------------------------------
    with tf.Graph().as_default():
        with tf.Session() as sess:
            if clf == 'rnn':
                classifier = rnn_clf(config)
            else:
                classifier = cnn_clf(config)

            # Training procedure
            global_step = tf.Variable(0, name='global_step', trainable=False)
            train_op = tf.train.AdamOptimizer(config.learning_rate).minimize(classifier.cost, global_step)

            # Summaries
            tf.summary.scalar('Loss', classifier.cost)
            tf.summary.scalar('Accuracy', classifier.accuracy)

            # Train summary
            train_summary_op = tf.summary.merge_all()
            train_summary_dir = os.path.join(outdir, 'summaries', 'train')
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Validation summary
            # valid_summary_op = tf.summary.merge([loss_summary, accuracy_summary])
            # valid_summary_dir = os.path.join(outdir, 'summaries', 'valid')
            # valid_summary_writer = tf.summary.FileWriter(valid_summary_dir, sess.graph)

            saver = tf.train.Saver(max_to_keep=20)

            sess.run(tf.global_variables_initializer())

            def run_step(input_data, is_training=True):
                """Run one step of the training process."""
                if clf == 'rnn':
                    input_x, input_y, length = input_data
                elif clf == 'cnn':
                    input_x, input_y = input_data

                fetches = {'step': global_step,
                           'cost': classifier.cost,
                           'accuracy': classifier.accuracy}
                feed_dict = {classifier.input_x: input_x,
                             classifier.input_y: input_y}

                if clf == 'rnn':
                    fetches['final_state'] = classifier.final_state
                    feed_dict[classifier.sequence_length] = length

                if is_training:
                    fetches['train_op'] = train_op
                    fetches['summaries'] = train_summary_op
                    feed_dict[classifier.keep_prob] = config.keep_prob
                else:
                    # fetches['summaries'] = valid_summary_op
                    feed_dict[classifier.keep_prob] = 1.0

                vars = sess.run(fetches, feed_dict)
                step = vars['step']
                cost = vars['cost']
                accuracy = vars['accuracy']
                if is_training:
                    summaries = vars['summaries']
                    train_summary_writer.add_summary(summaries, step)

                return cost, accuracy

            print('Start training ...')

            for i in range(config.num_epochs):
                # Mini-batch iterator
                train_data = data_helper.batch_iter(x_train, y_train, config.batch_size, max_length, clf=clf)
                valid_data = data_helper.batch_iter(x_valid, y_valid, 1, max_length, clf=clf)
                # test_data = data_helper.batch_iter(x_test, y_test, 1, max_length, clf=clf)

                train_step = 0
                valid_step = 0
                # test_step = 0
                tot_train_cost = 0
                tot_valid_cost = 0
                # tot_test_cost = 0
                tot_train_accuracy = 0
                tot_valid_accuracy = 0
                # tot_test_accuracy = 0

                for train_input in train_data:
                    train_cost, train_accuracy = run_step(train_input, is_training=True)

                    tot_train_accuracy += train_accuracy
                    tot_train_cost += train_cost

                    train_step += 1

                    if train_step % 100 == 0:
                        print('Epoch: {}, Batch: {}, Loss: {}, Accuracy: {}'.format(i,
                                                                                    train_step,
                                                                                    train_cost,
                                                                                    train_accuracy))

                for valid_input in valid_data:
                    valid_cost, valid_accuracy = run_step(valid_input, is_training=False)

                    tot_valid_accuracy += valid_accuracy
                    tot_valid_cost += valid_cost

                    valid_step += 1

                # for test_input in test_data:
                #     test_cost, test_accuracy = run_step(test_input, is_training=False)
                #
                #     tot_test_accuracy += test_accuracy
                #     tot_test_cost += test_cost
                #
                #     test_step += 1

                print('=============================================')
                print('Epoch: {}, Train loss: {}, Train accuracy: {}'.format(i,
                                                                             tot_train_cost / train_step,
                                                                             tot_train_accuracy / train_step))
                print('Epoch: {}, Valid loss: {}, Valid accuracy: {}'.format(i,
                                                                             tot_valid_cost / valid_step,
                                                                             tot_valid_accuracy / valid_step))
                # print('Epoch: {}, Test loss: {}, Test accuracy: {}'.format(i,
                #                                                            tot_test_cost / test_step,
                #                                                            tot_test_accuracy / test_step))
                print('=============================================')

                # Save the model at every 5 epochs
                if i % 5 == 0:
                    saver.save(sess, os.path.join(outdir, 'model/clf'), i)
                    print('Model saved')

if __name__ == '__main__':
    train(clf='cnn')
