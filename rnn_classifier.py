# -*- coding: utf-8 -*-
import tensorflow as tf

class rnn_clf(object):
    """"
    A RNN classifier for text classification
    """
    def __init__(self, config):
        self.num_classes = config.num_classes
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.l2_reg_lambda = config.l2_reg_lambda

        # Placeholders
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[])
        self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, None])
        self.input_y = tf.placeholder(dtype=tf.int64, shape=[None])
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=[])
        self.sequence_length = tf.placeholder(dtype=tf.int32, shape=[None])

        # L2 loss
        self.l2_loss = tf.constant(0.0)

        # LSTM Cell
        cell = tf.contrib.rnn.LSTMCell(self.hidden_size,
                                       forget_bias=1.0,
                                       state_is_tuple=True,
                                       reuse=tf.get_variable_scope().reuse)
        # Add dropout to cell output
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        # Stacked LSTMs
        cell = tf.contrib.rnn.MultiRNNCell([cell]*self.num_layers, state_is_tuple=True)

        self._initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)

        # Word embedding
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            # embedding = tf.Variable(tf.random_uniform([self.vocab_size, self.hidden_size], -1.0, 1.0),
            #                         name='embedding')
            # better performance
            embedding = tf.get_variable('embedding',
                                        shape=[self.vocab_size, self.hidden_size],
                                        dtype=tf.float32)
            print(embedding.shape)
            inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        print(inputs.shape)

        # Input dropout
        inputs = tf.nn.dropout(inputs, keep_prob=self.keep_prob)

        # Dynamic LSTM
        with tf.variable_scope('LSTM'):
            outputs, state = tf.nn.dynamic_rnn(cell,
                                               inputs=inputs,
                                               initial_state=self._initial_state,
                                               sequence_length=self.sequence_length)

        self.final_state = state

        # Softmax output layer
        with tf.name_scope('softmax'):
            softmax_w = tf.get_variable('softmax_w', shape=[self.hidden_size, self.num_classes], dtype=tf.float32)
            softmax_b = tf.get_variable('softmax_b', shape=[self.num_classes], dtype=tf.float32)

            # L2 regularization for output layer
            self.l2_loss += tf.nn.l2_loss(softmax_w)
            self.l2_loss += tf.nn.l2_loss(softmax_b)

            self.logits = tf.matmul(self.final_state[self.num_layers - 1].h, softmax_w) + softmax_b
            predictions = tf.nn.softmax(self.logits)
            self.predictions = tf.argmax(predictions, 1)

        # Loss
        with tf.name_scope('loss'):
            tvars = tf.trainable_variables()

            # L2 regularization for LSTM weights
            for tv in tvars:
                if 'kernel' in tv.name:
                    self.l2_loss += tf.nn.l2_loss(tv)

            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y,
                                                                  logits=self.logits)
            self.cost = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss

        # Accuracy
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, self.input_y)
            self.correct_num = tf.reduce_sum(tf.cast(correct_predictions, tf.float32))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')
