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

# Please download langconv.py and zh_wiki.py first
# langconv.py and zh_wiki.py are used for converting between languages
try:
    import langconv
except ImportError as e:
    error = "Please download langconv.py and zh_wiki.py at "
    error += "https://github.com/skydark/nstools/tree/master/zhtools."
    print(str(e) + ': ' + error)
    sys.exit()

def load_data(file_path, sw_path, test_file_path=None, language='ch', save_path=None, vocab_size=1000):
    """
    Build dataset for mini-batch iterator
    :param file_path: Data file path
    :param sw_path: Stop word file path
    :param test_file_path: Test data file path
    :param language: 'ch' for Chinese and 'en' for English
    :param save_path: the path to save the mapping result
    :param vocab_size: expected vocabulary size
    :return data: a list of sentences. each sentence is a vector of integers
    :return labels: a list of labels
    :return idx_2_w: a vocabulary index
    :return len(idx_2_w): true vocabulary size
    :return max_length: the length of the longest sentence
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        incsv = csv.reader(f)
        header = next(incsv)  # Headers
        label_idx = header.index('label')
        content_idx = header.index('content')

        text = []
        labels = []
        words = []

        sw = _stop_words(sw_path)

        for line in incsv:
            sent = line[content_idx].strip()
            if language == 'ch':
                sent = _tradition_2_simple(sent)
            elif language == 'en':
                sent = sent.lower()

            sent = _clean_data(sent, sw, language=language)

            if len(sent) < 1:
                continue

            word_list = _word_segmentation(sent, language)
            words.extend(word_list)
            text.append(word_list)
            labels.append(line[label_idx])

    start = time.time()
    print("Building dataset....")
    count = [['<PAD>', -1], ['<UNK>', 0]]
    count.extend(collections.Counter(words).most_common(vocab_size - 2))
    words, _ = zip(*count)
    words = list(words)
    del count  # Release memory

    # Map words to indices
    w_2_idx = dict(zip(words, range(len(words))))
    idx_2_w = dict(zip(w_2_idx.values(), w_2_idx.keys()))

    data = []
    for sentence in text:
        temp = []
        for word in sentence:
            if word in words:
                temp.append(w_2_idx[word])
            else:
                temp.append(w_2_idx['<UNK>'])
        data.append(temp)
    del text  # Release memory

    if test_file_path is not None:
        test_data, test_labels = load_test_data(test_file_path, sw, w_2_idx, language)

    end = time.time()
    runtime = end - start

    print('Dataset has been built successfully.')
    print('Run time: ', runtime)
    print('--------- Summary of the Dataset ---------')
    print('Vocabulary size: ', len(w_2_idx))
    print('Number of sentences: ', len(data))
    print('Words to indices: ', sorted(w_2_idx.items(), key=lambda x: x[1])[:5])
    print('-------------------------------------------')

    if save_path is not None:
        if os.path.isfile(save_path):
            raise RuntimeError('the save path should be a dir')
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        map_path = os.path.join(save_path, 'maps.txt')
        with open(map_path, 'w', encoding='utf-8') as outf:
            for idx in sorted(idx_2_w.items(), key=lambda x: x[0]):
                outstr = '{}\t{}'.format(idx[0], idx_2_w[idx[0]])
                outf.write(outstr)
                outf.write('\n')

    max_length = max(map(len, data))

    if test_file_path is not None:
        return data, labels, idx_2_w, len(idx_2_w), max_length, test_data, test_labels
    else:
        return data, labels, idx_2_w, len(idx_2_w), max_length

def load_test_data(file_path, sw, w_2_idx, language='ch'):
    data = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        incsv = csv.reader(f)
        header = next(incsv)
        label_idx = header.index('label')
        content_idx = header.index('content')

        for line in incsv:
            sent_2_indices = []
            sent = line[content_idx].strip()
            if language == 'ch':
                sent = _tradition_2_simple(sent)
            elif language == 'en':
                sent = sent.lower()

            sent = _clean_data(sent, sw, language=language)

            if len(sent) < 1:
                continue

            word_list = _word_segmentation(sent, language)
            for word in word_list:
                if word in w_2_idx:
                    sent_2_indices.append(w_2_idx[word])
                else:
                    sent_2_indices.append(w_2_idx['<UNK>'])
            data.append(sent_indices)
            labels.append(line[label_idx])

    return data, labels



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


def restore_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as mapf:
        w_2_idx = {}
        idx_2_w = {}
        for line in mapf:
            idx, word = line.strip().split('\t')
            w_2_idx[word] = int(idx)
            idx_2_w[int(idx)] = word
    return w_2_idx, idx_2_w


# --------------- Private Methods ---------------

# Convert Traditional Chinese to Simplified Chinese
def _tradition_2_simple(sent):
    return langconv.Converter('zh-hans').convert(sent)


def _word_segmentation(sent, language):
    if language == 'ch':
        return list(jieba.cut(sent, cut_all=False, HMM=True))
    elif language == 'en':
        return nltk.word_tokenize(sent)


def _stop_words(path):
    with open(path, 'r', encoding='utf-8') as f:
        sw = list()
        for line in f:
            sw.append(line.strip())

    return set(sw)


# Delete stop words using custom dictionary
def _clean_data(sent, sw, language='ch'):
    if language == 'ch':
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
    data, labels, idx_2_w_a, _, max_length = load_data('data.csv', 'stop_words_ch.txt', test_file_path='test.csv' language='ch', save_path='data')
    print(max_length)
