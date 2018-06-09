# -*- coding: utf-8 -*-
import tensorflow as tf

class rnn_clf(object):
    """"
    LSTM and Bi-LSTM classifiers for text classification
    """
    def __init__(self, config):
        self.num_classes = config.num_classes
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.l2_reg_lambda = config.l2_reg_lambda

        # Placeholders
        self.batch_size = tf.placeholder(dtype=tf.int32, shape=[], name='batch_size')
        self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_x')
        self.input_y = tf.placeholder(dtype=tf.int64, shape=[None], name='input_y')
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
        self.sequence_length = tf.placeholder(dtype=tf.int32, shape=[None], name='sequence_length')

        # L2 loss
        self.l2_loss = tf.constant(0.0)

        # Word embedding
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            embedding = tf.get_variable('embedding',
                                        shape=[self.vocab_size, self.hidden_size],
                                        dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        # Input dropout
        self.inputs = tf.nn.dropout(inputs, keep_prob=self.keep_prob)

        # LSTM
        if config.clf == 'lstm':
            self.final_state = self.normal_lstm()
        else:
            self.final_state = self.bi_lstm()

        # Softmax output layer
        with tf.name_scope('softmax'):
            # softmax_w = tf.get_variable('softmax_w', shape=[self.hidden_size, self.num_classes], dtype=tf.float32)
            if config.clf == 'lstm':
                softmax_w = tf.get_variable('softmax_w', shape=[self.hidden_size, self.num_classes], dtype=tf.float32)
            else:
                softmax_w = tf.get_variable('softmax_w', shape=[2 * self.hidden_size, self.num_classes], dtype=tf.float32)
            softmax_b = tf.get_variable('softmax_b', shape=[self.num_classes], dtype=tf.float32)

            # L2 regularization for output layer
            self.l2_loss += tf.nn.l2_loss(softmax_w)
            self.l2_loss += tf.nn.l2_loss(softmax_b)

            # self.logits = tf.matmul(self.final_state[self.num_layers - 1].h, softmax_w) + softmax_b
            if config.clf == 'lstm':
                self.logits = tf.matmul(self.final_state[self.num_layers - 1].h, softmax_w) + softmax_b
            else:
                self.logits = tf.matmul(self.final_state, softmax_w) + softmax_b
            predictions = tf.nn.softmax(self.logits)
            self.predictions = tf.argmax(predictions, 1, name='predictions')

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

    def normal_lstm(self):
        # LSTM Cell
        cell = tf.contrib.rnn.LSTMCell(self.hidden_size,
                                       forget_bias=1.0,
                                       state_is_tuple=True,
                                       reuse=tf.get_variable_scope().reuse)
        # Add dropout to cell output
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        # Stacked LSTMs
        cell = tf.contrib.rnn.MultiRNNCell([cell] * self.num_layers, state_is_tuple=True)

        self._initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)

        # Dynamic LSTM
        with tf.variable_scope('LSTM'):
            outputs, state = tf.nn.dynamic_rnn(cell,
                                               inputs=self.inputs,
                                               initial_state=self._initial_state,
                                               sequence_length=self.sequence_length)

        final_state = state

        return final_state


    def bi_lstm(self):
        cell_fw = tf.contrib.rnn.LSTMCell(self.hidden_size,
                                          forget_bias=1.0,
                                          state_is_tuple=True,
                                          reuse=tf.get_variable_scope().reuse)
        cell_bw = tf.contrib.rnn.LSTMCell(self.hidden_size,
                                          forget_bias=1.0,
                                          state_is_tuple=True,
                                          reuse=tf.get_variable_scope().reuse)

        # Add dropout to cell output
        cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.keep_prob)
        cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.keep_prob)

        # Stacked LSTMs
        cell_fw = tf.contrib.rnn.MultiRNNCell([cell_fw] * self.num_layers, state_is_tuple=True)
        cell_bw = tf.contrib.rnn.MultiRNNCell([cell_bw] * self.num_layers, state_is_tuple=True)

        self._initial_state_fw = cell_fw.zero_state(self.batch_size, dtype=tf.float32)
        self._initial_state_bw = cell_bw.zero_state(self.batch_size, dtype=tf.float32)

        # Dynamic Bi-LSTM
        with tf.variable_scope('Bi-LSTM'):
            _, state = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                       cell_bw,
                                                       inputs=self.inputs,
                                                       initial_state_fw=self._initial_state_fw,
                                                       initial_state_bw=self._initial_state_bw,
                                                       sequence_length=self.sequence_length)

        state_fw = state[0]
        state_bw = state[1]
        output = tf.concat([state_fw[self.num_layers - 1].h, state_bw[self.num_layers - 1].h], 1)

        return output
