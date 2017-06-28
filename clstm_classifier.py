# -*- coding: utf-8 -*-
import tensorflow as tf


class config():
    max_length = 10
    num_classes = 3
    batch_size = 20
    vocab_size = 100
    embedding_size = 200
    filter_sizes = [3]
    num_filters = 3
    hidden_size = 10
    num_layers = 1
    l2_reg_lambda = 0.0


class clstm_clf(object):
    """
    A C-LSTM classifier for text classification
    """
    def __init__(self, config):
        self.sequence_length = config.max_length
        self.num_classes = config.num_classes
        self.batch_size = config.batch_size
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        self.filter_sizes = config.filter_sizes
        self.num_filters = config.num_filters
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.l2_reg_lambda = config.l2_reg_lambda

        # Placeholders
        self.input_x = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.sequence_length])
        self.input_y = tf.placeholder(dtype=tf.int64, shape=[self.batch_size])
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=[])

        # L2 loss
        self.l2_loss = tf.constant(0.0)

        # Word embedding
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            embedding = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                                    name="embedding")
            embed = tf.nn.embedding_lookup(embedding, self.input_x)
            inputs = tf.expand_dims(embed, -1)

        # Input dropout
        inputs = tf.nn.dropout(inputs, keep_prob=self.keep_prob)

        cnn_outputs = []
        # Convolutional layer with different lengths of filters in parallel
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope('conv-%s' % filter_size):
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                print(filter_shape)
                W = tf.get_variable('weights', filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
                b = tf.get_variable('biases', [self.num_filters], initializer=tf.constant_initializer(0.0))

                # Convolution
                conv = tf.nn.conv2d(inputs,
                                    W,
                                    strides=[1, 1, 1, 1],
                                    padding='VALID',
                                    name='conv')
                # Activation function
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                # TODO reshape the CNN output tensors to proper shape.

                cnn_outputs.append(h)

        # LSTM cell
        cell = tf.contrib.rnn.LSTMCell(self.hidden_size,
                                       forget_bias=1.0,
                                       state_is_tuple=True,
                                       reuse=tf.get_variable_scope().reuse)
        # Add dropout to LSTM cell
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        # Stacked LSTMs
        cell = tf.contrib.rnn.MultiRNNCell([cell]*self.num_layers, state_is_tuple=True)

        self._initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)

        # Dynamic LSTM
        with tf.variable_scope('LSTM'):
            outputs, state = tf.nn.dynamic_rnn(cell,
                                              cnn_outputs,
                                              initial_state=self._initial_state)
            self.final_state = state

        # Softmax output layer
        with tf.name_scope('softmax'):
            softmax_w = tf.get_variable('softmax_w', shape=[self.hidden_size, self.num_classes], dtype=tf.float32)
            softmax_b = tf.get_variable('softmax_b', shape=[self.num_classes], dtype=tf.float32)

            # L2 regularization for output layer
            self.l2_loss += tf.nn.l2_loss(softmax_w)
            self.l2_loss += tf.nn.l2_loss(softmax_b)

            # logits
            self.logits = tf.matmul(self.final_state[self.num_layers - 1].h, softmax_w) + softmax_b
            predictions = tf.nn.softmax(self.logits)
            self.predictions = tf.argmax(predictions, 1)

        # Loss
        with tf.name_scope('loss'):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            self.cost = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss

        # Accuracy
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, self.input_y)
            self.correct_num = tf.reduce_sum(tf.cast(correct_predictions, tf.float32))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')


clstm = clstm_clf(config)