# -*- coding: utf-8 -*-
import tensorflow as tf
import data_helper


class config():
    num_classes = 2
    batch_size = 32
    # vocab_size = 100
    embedding_size = 256
    filter_sizes = '3'
    num_filters = 256
    hidden_size = 256
    num_layers = 3
    l2_reg_lambda = 0.0


class clstm_clf(object):
    """
    A C-LSTM classifier for text classification
    """
    def __init__(self, config):
        self.max_length = config.max_length
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
        self.input_x = tf.placeholder(dtype=tf.int32, shape=[self.batch_size, self.max_length])
        self.input_y = tf.placeholder(dtype=tf.int64, shape=[self.batch_size])
        self.keep_prob = tf.placeholder(dtype=tf.float32, shape=[])
        self.sequence_length = tf.placeholder(dtype=tf.int32, shape=[self.batch_size])

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

        # Convolutional layer with different lengths of filters in parallel
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope('conv-%s' % filter_size):
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                W = tf.get_variable('weights', filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
                b = tf.get_variable('biases', [self.num_filters], initializer=tf.constant_initializer(0.0))

                # Convolution
                conv = tf.nn.conv2d(inputs,
                                    W,
                                    strides=[1, 1, 1, 1],
                                    padding='VALID',
                                    name='conv')
                # Activation function
                self.h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                # Reshape the CNN output tensors to proper shape.
                self.h_reshape = tf.squeeze(self.h)


        # LSTM cell
        cell = tf.contrib.rnn.LSTMCell(self.hidden_size,
                                       forget_bias=1.0,
                                       state_is_tuple=True)
        # Add dropout to LSTM cell
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)

        # Stacked LSTMs
        cell = tf.contrib.rnn.MultiRNNCell([cell]*self.num_layers, state_is_tuple=True)

        self._initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)

        # Feed the CNN outputs to LSTM network
        with tf.variable_scope('LSTM'):
            outputs, state = tf.nn.dynamic_rnn(cell,
                                               self.h_reshape,
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

        tf.summary.scalar('Loss', self.cost)
        tf.summary.scalar('Accuracy', self.accuracy)

        self.summary_op = tf.summary.merge_all()

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        self.train_op = optimizer.minimize(self.cost, global_step=self.global_step)


data, labels, w_2_idx = data_helper.load_data(file_path='benchmark.csv', sw_path='stop_words_ch.txt', language='en', shuffle=True)
config.max_length = max(map(len, data))
batches = data_helper.batch_iter(data=data, labels=labels, w_2_idx=w_2_idx, batch_size=config.batch_size, num_epochs=50)

with tf.Session() as sess:
    config.vocab_size = len(w_2_idx)
    clstm = clstm_clf(config)

    sess.run(tf.global_variables_initializer())

    summary_writer = tf.summary.FileWriter('summary/', sess.graph)
    for batch in batches:
        xdata, ydata, length = batch
        length = length - 2
        feed_dict = {clstm.input_x: xdata,
                     clstm.input_y: ydata,
                     clstm.keep_prob: 0.5,
                     clstm.sequence_length: length}
        step, acc, loss, summaries, _ = sess.run([clstm.global_step, clstm.accuracy, clstm.cost, clstm.summary_op, clstm.train_op], feed_dict)
        summary_writer.add_summary(summaries, step)
        print("step: {}, loss: {:g}, accuracy: {:g}".format(step, loss, acc))
