# -*- coding: utf-8 -*-
import csv
import collections
import re
import sys
import time
import os
import json

import numpy as np
import jieba

# Please download langconv.py and zh_wiki.py first
# langconv.py and zh_wiki.py are used for converting between languages
try:
    import langconv
except ImportError as e:
    error = "Please download langconv.py and zh_wiki.py at "
    error += "https://github.com/skydark/nstools/tree/master/zhtools."
    print(str(e) + ': ' + error)
    sys.exit()

# Load data from csv file
# file_path: csv file path
# sw_path: stop word file path
def load_data(file_path, sw_path, save_path=None, vocab_size=10000):
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
            sent = _tradition_2_simple(line[content_idx].strip())
            sent = _clean_data(sent, sw)

            if len(sent) < 1:
                continue

            word_list = _word_segmentation(sent)
            words.extend(word_list)
            text.append(word_list)
            labels.append(line[label_idx])

    start = time.time()
    print("Building dataset....")
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocab_size - 1))
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
                temp.append(w_2_idx['UNK'])
        data.append(temp)
    del text  # Release memory

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

    return data, labels, idx_2_w, len(idx_2_w)


def batch_iter(data, labels, batch_size, shuffle=True):
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
        max_length = max(map(len, batch))  # Train data length

        xdata = np.full((batch_size, max_length), 0, np.int32)
        ydata = np.full(batch_size, 0, np.int64)
        for row in range(batch_size):
            xdata[row, :len(batch[row])] = batch[row]
            if int(label[row]) < 0:
                ydata[row] = 2
            else:
                ydata[row] = int(label[row])

        yield (xdata, ydata, length)

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


def _word_segmentation(sent):
    return list(jieba.cut(sent, cut_all=False, HMM=True))


def _stop_words(path):
    with open(path, 'r', encoding='utf-8') as f:
        sw = list()
        for line in f:
            sw.append(line.strip())

    return set(sw)


# Delete stop words using custom dictionary
def _clean_data(sent, sw):
    sent = re.sub('\s+', '', sent)
    sent = re.sub('！+', '！', sent)
    sent = re.sub('？+', '！', sent)
    sent = re.sub('。+', '。', sent)
    sent = re.sub('，+', '，', sent)
    sent = "".join([word for word in sent if word not in sw])

    return sent

if __name__ == '__main__':
    # Tiny example for test
    data, labels, idx_2_w_a, _ = load_data('test.csv', 'stop_words_ch.txt', save_path='data')
    for data in batch_iter(data, labels, batch_size=1):
        sentence, label, length = data
        str = ''
        for word in sentence[0]:
            str += idx_2_w_a[word]
        print(str, label, length)
    w_2_idx, idx_2_w_b = restore_data('data/maps.txt')
    assert idx_2_w_a == idx_2_w_b
