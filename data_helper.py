# -*- coding: utf-8 -*-
import re
import os
import sys
import csv
import time
import json
import collections

import nltk
import jieba
import numpy as np
from tensorflow.contrib import learn

# Please download langconv.py and zh_wiki.py first
# langconv.py and zh_wiki.py are used for converting between languages
try:
    import langconv
except ImportError as e:
    error = "Please download langconv.py and zh_wiki.py at "
    error += "https://github.com/skydark/nstools/tree/master/zhtools."
    print(str(e) + ': ' + error)
    sys.exit()


def load_data(file_path, sw_path, min_frequency=0, language='ch', vocab_processor=None):
    """
    Build dataset for mini-batch iterator
    :param file_path: Data file path
    :param sw_path: Stop word file path
    :param language: 'ch' for Chinese and 'en' for English
    :param min_frequency: the minimal frequency of words to keep
    :return data: a list of sentences. each sentence is a vector of integers
    :return labels: a list of labels
    :return vocab_processor: Tensorflow VocabularyProcessor object
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        print('Building dataset ...')
        start = time.time()
        incsv = csv.reader(f)
        header = next(incsv)  # Header
        label_idx = header.index('label')
        content_idx = header.index('content')

        labels = []
        sentences = []

        sw = _stop_words(sw_path)

        for line in incsv:
            sent = line[content_idx].strip()

            if language == 'ch':
                sent = _tradition_2_simple(sent)  # Convert traditional Chinese to simplified Chinese
            elif language == 'en':
                sent = sent.lower()

            sent = _clean_data(sent, sw, language=language)  # Remove stop words and special characters

            if len(sent) < 1:
                continue

            sent = _word_segmentation(sent, language)
            sentences.append(sent)
            labels.append(line[label_idx])

    max_length = max(map(len, [sent.strip().split(' ') for sent in sentences]))

    # Extract vocabulary from sentences and map words to indices
    if vocab_processor is None:
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_length, min_frequency=min_frequency)
        data = np.array(list(vocab_processor.fit_transform(sentences)))
    else:
        data = np.array(list(vocab_processor.transform(sentences)))

    # Sentence vector
    end = time.time()

    print('Dataset has been built successfully.')
    print('Run time: {}'.format(end - start))
    print('Number of sentences: {}'.format(len(data)))
    print('Vocabulary size: {}'.format(len(vocab_processor.vocabulary_._mapping)))
    print('Max document length: {}'.format(vocab_processor.max_document_length))

    return data, labels, vocab_processor


def batch_iter(data, labels, batch_size, max_length=0, clf='rnn', shuffle=True):
    """
    A mini-batch iterator to generate mini-batches for training neural network
    :param data: a list of sentences. each sentence is a vector of integers
    :param labels: a list of labels
    :param batch_size: the size of mini-batch
    :param max_length: the length of the longest sentence in the dataset
    :param clf: the type of neural network classifier
    :param shuffle: whether to shuffle the data
    :return: a mini-batch iterator
    """
    data = np.array(data)
    labels = np.array(labels)
    data_size = len(data)
    epoch_length = data_size // batch_size

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        data = data[shuffle_indices]
        labels = labels[shuffle_indices]

    for i in range(epoch_length):
        start_index = i * batch_size
        end_index = start_index + batch_size

        batch = data[start_index: end_index]
        label = labels[start_index: end_index]
        length = np.asarray(list(map(len, batch)))  # Convert list to numpy array

        if clf == 'rnn':
            max_length = max(map(len, batch))

        xdata = np.full((batch_size, max_length), 0, np.int32)  # Zero padding
        ydata = np.full(batch_size, 0, np.int64)
        for row in range(batch_size):
            xdata[row, :len(batch[row])] = batch[row]
            if int(label[row]) < 0:
                ydata[row] = 2
            else:
                ydata[row] = int(label[row])

        if clf == 'rnn':
            yield (xdata, ydata, length)
        if clf == 'cnn':
            yield (xdata, ydata)

# --------------- Private Methods ---------------

def _tradition_2_simple(sent):
    """ Convert Traditional Chinese to Simplified Chinese """
    return langconv.Converter('zh-hans').convert(sent)


def _word_segmentation(sent, language):
    """ Tokenizer """
    if language == 'ch':
        sent = ' '.join(list(jieba.cut(sent, cut_all=False, HMM=True)))
    elif language == 'en':
        sent = ' '.join(nltk.word_tokenize(sent))

    return re.sub(r'\s+', ' ', sent)


def _stop_words(path):
    with open(path, 'r', encoding='utf-8') as f:
        sw = list()
        for line in f:
            sw.append(line.strip())

    return set(sw)


def _clean_data(sent, sw, language='ch'):
    """ Remove special characters and stop words """
    if language == 'ch':
        sent = re.sub(r"[^\u4e00-\u9fa5A-z0-9]", " ", sent)
        sent = re.sub('\s+', ' ', sent)
        sent = re.sub('！+', '！', sent)
        sent = re.sub('？+', '！', sent)
        sent = re.sub('。+', '。', sent)
        sent = re.sub('，+', '，', sent)
    if language == 'en':
        sent = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", sent)
        sent = re.sub(r"\'s", " \'s", sent)
        sent = re.sub(r"\'ve", " \'ve", sent)
        sent = re.sub(r"n\'t", " n\'t", sent)
        sent = re.sub(r"\'re", " \'re", sent)
        sent = re.sub(r"\'d", " \'d", sent)
        sent = re.sub(r"\'ll", " \'ll", sent)
        sent = re.sub(r",", " , ", sent)
        sent = re.sub(r"!", " ! ", sent)
        sent = re.sub(r"\(", " \( ", sent)
        sent = re.sub(r"\)", " \) ", sent)
        sent = re.sub(r"\?", " \? ", sent)
        sent = re.sub(r"\s{2,}", " ", sent)
    sent = "".join([word for word in sent if word not in sw])

    return sent

if __name__ == '__main__':
    # Tiny example for test
    data, labels, vocab_processor = load_data(file_path='test.csv', sw_path='stop_words_ch.txt')
    print(data)
    vocab_processor.save('vocab')
    vocab_pro = learn.preprocessing.VocabularyProcessor.restore('vocab')
    data, labels, _ = load_data(file_path='test1.csv', sw_path='stop_words_ch.txt', vocab_processor=vocab_pro)
    print(data)
