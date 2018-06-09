# -*- coding: utf-8 -*-
import tensorflow as tf

class cnn_clf(object):
    """
    A CNN classifier for text classification
    """
    def __init__(self, config):
        self.max_length = config.max_length
        self.num_classes = config.num_classes
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        self.filter_sizes = list(map(int, config.filter_sizes.split(",")))
        self.num_filters = config.num_filters
        self.l2_reg_lambda = config.l2_reg_lambda

        # Placeholders
        self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, self.max_length], name='input_x')
        self.input_y = tf.placeholder(dtype=tf.int64, shape=[None], name='input_y')
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')

        # L2 loss
        self.l2_loss = tf.constant(0.0)

        # Word embedding
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            embedding = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                                    name="embedding")
            embed = tf.nn.embedding_lookup(embedding, self.input_x)
            inputs = tf.expand_dims(embed, -1)

        # Convolution & Maxpool
        pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                # Convolution
                filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                W = tf.get_variable("weights", filter_shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
                b = tf.get_variable("biases", [self.num_filters], initializer=tf.constant_initializer(0.0))

                conv = tf.nn.conv2d(inputs,
                                    W,
                                    strides=[1, 1, 1, 1],
                                    padding='VALID',
                                    name='conv')
                # Activation function
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                # Maxpool
                pooled = tf.nn.max_pool(h,
                                        ksize=[1, self.max_length - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1],
                                        padding='VALID',
                                        name='pool')
                pooled_outputs.append(pooled)

        num_filters_total = self.num_filters * len(self.filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        # Add dropout
        h_drop = tf.nn.dropout(h_pool_flat, keep_prob=self.keep_prob)

        # Softmax
        with tf.name_scope('softmax'):
            softmax_w = tf.Variable(tf.truncated_normal([num_filters_total, self.num_classes], stddev=0.1), name='softmax_w')
            softmax_b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name='softmax_b')

            # Add L2 regularization to output layer
            self.l2_loss += tf.nn.l2_loss(softmax_w)
            self.l2_loss += tf.nn.l2_loss(softmax_b)

            self.logits = tf.matmul(h_drop, softmax_w) + softmax_b
            predictions = tf.nn.softmax(self.logits)
            self.predictions = tf.argmax(predictions, 1, name='predictions')

        # Loss
        with tf.name_scope('loss'):
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            # Add L2 losses
            self.cost = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss

        # Accuracy
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, self.input_y)
            self.correct_num = tf.reduce_sum(tf.cast(correct_predictions, tf.float32))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')
