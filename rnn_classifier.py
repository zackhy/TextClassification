# -*- coding: utf-8 -*-
import tensorflow as tf

class rnn_clf(object):
    def __init__(self,
                 num_classes, # int
                 batch_size,  # int
                 vocab_size,  # int
                 embedding_size,  # int
                 hidden_size,  # int
                 num_layers,  # int
                 learning_rate,  # float
                 keep_prob,  # float
                 l2_reg_lambda=0.0,  # float
                 is_training=True,  # boolean
                 ):

        self.input_x = tf.placeholder(dtype=tf.int32, shape=[batch_size, None])
        self.input_y = tf.placeholder(dtype=tf.int64, shape=batch_size)
        self.sequence_length = tf.placeholder(dtype=tf.int32, shape=batch_size)

        # L2 loss
        self.l2_loss = tf.constant(0.0)

        # LSTM Cell
        cell = tf.contrib.rnn.LSTMCell(hidden_size,
                                       forget_bias=1.0,
                                       state_is_tuple=True,
                                       reuse=tf.get_variable_scope().reuse)
        # Add dropout to cell output
        if is_training and keep_prob < 1:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)

        # Stacked LSTMs
        cell = tf.contrib.rnn.MultiRNNCell([cell]*num_layers, state_is_tuple=True)

        self._initial_state = cell.zero_state(batch_size, dtype=tf.float32)

        # Word embedding
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                                    name='embedding')
            inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        # Input dropout
        if is_training and keep_prob < 1:
            inputs = tf.nn.dropout(inputs, keep_prob=keep_prob)

        # Dynamic LSTM
        with tf.variable_scope('LSTM'):
            outputs, state = tf.nn.dynamic_rnn(cell,
                                               inputs=inputs,
                                               initial_state=self._initial_state,
                                               sequence_length=self.sequence_length)

        # self.output = state[num_layers - 1].h
        self.final_state = state

        # Softmax output layer
        with tf.name_scope('Softmax'):
            softmax_w = tf.get_variable('softmax_w', shape=[hidden_size, num_classes], dtype=tf.float32)
            softmax_b = tf.get_variable('softmax_b', shape=[num_classes], dtype=tf.float32)

            # L2 regularization
            self.l2_loss += tf.nn.l2_loss(softmax_w)
            self.l2_loss += tf.nn.l2_loss(softmax_b)

            self.logits = tf.matmul(self.final_state[num_layers - 1].h, softmax_w) + softmax_b
            predictions = tf.nn.softmax(self.logits)
            self.predictions = tf.argmax(predictions, 1)

        # Loss
        with tf.name_scope('Loss'):
            tvars = tf.trainable_variables()
            for tv in tvars:
                if 'kernel' in tv.name:
                    self.l2_loss += tf.nn.l2_loss(tv)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y,
                                                                  logits=self.logits)
            self.cost = tf.reduce_mean(loss) + l2_reg_lambda * self.l2_loss

        # Accuracy
        with tf.name_scope('Accuracy'):
            correct_predictions = tf.equal(self.predictions, self.input_y)
            self.correct_num = tf.reduce_sum(tf.cast(correct_predictions, tf.float32))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')

        if not is_training:
            return

        # Optimizer
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
