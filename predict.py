# -*- coding: utf-8 -*-
import os
import csv
import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.contrib import learn
from sklearn.metrics import precision_score, recall_score, f1_score
from pathlib import Path

from . import data_helper

def predict(run_dir, checkpoint_name, test_data_file_path, input_batch_size):
    """Predict the output for the test_data_file. Return the positive sents file."""
    test_data_file_path = Path(test_data_file_path)
    # Restore parameters
    with open(os.path.join(run_dir, 'params.pkl'), 'rb') as f:
        params = pkl.load(f, encoding='bytes')

    # Restore vocabulary processor
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(os.path.join(run_dir, 'vocab'))

    # Load test data
    data, labels, lengths, _ = data_helper.load_data(file_path=test_data_file_path,
                                                     sw_path=params.get('stop_word_file'),
                                                     min_frequency=params.get('min_frequency'),
                                                     max_length=params.get('max_length'),
                                                     language=params.get('language'),
                                                     vocab_processor=vocab_processor,
                                                     shuffle=False)

    # Restore graph
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        # Restore metagraph
        saver = tf.train.import_meta_graph('{}.meta'.format(os.path.join(run_dir, 'model', checkpoint_name)))
        # Restore weights
        saver.restore(sess, os.path.join(run_dir, 'model', checkpoint_name))

        # Get tensors
        input_x = graph.get_tensor_by_name('input_x:0')
        input_y = graph.get_tensor_by_name('input_y:0')
        keep_prob = graph.get_tensor_by_name('keep_prob:0')
        predictions = graph.get_tensor_by_name('softmax/predictions:0')
        accuracy = graph.get_tensor_by_name('accuracy/accuracy:0')

        # Generate batches
        batches = data_helper.batch_iter(data, labels, lengths, input_batch_size, 1)

        num_batches = int(len(data)/input_batch_size) + 1
        all_predictions = []
        sum_accuracy = 0

        # Test
        for batch in batches:
            x_test, y_test, x_lengths = batch
            if params['clf'] == 'cnn':
                feed_dict = {input_x: x_test, input_y: y_test, keep_prob: 1.0}
                batch_predictions, batch_accuracy = sess.run([predictions, accuracy], feed_dict)
            else:
                batch_size = graph.get_tensor_by_name('batch_size:0')
                sequence_length = graph.get_tensor_by_name('sequence_length:0')
                feed_dict = {input_x: x_test, input_y: y_test, batch_size: input_batch_size, sequence_length: x_lengths, keep_prob: 1.0}

                batch_predictions, batch_accuracy = sess.run([predictions, accuracy], feed_dict)

            sum_accuracy += batch_accuracy
            all_predictions = np.concatenate([all_predictions, batch_predictions])

        final_accuracy = sum_accuracy / num_batches

    with open(test_data_file_path, 'r') as f:
        reader = csv.reader(f)
        sents = [row[1] for row in reader]

    # Save all predictions.
    preds_output_file_path = test_data_file_path.with_suffix('.sent_preds')
    with open(preds_output_file_path, 'w', encoding='utf-8', newline='') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(['True class', 'Prediction', 'Sentence'])
        for i in range(len(all_predictions)):
            csvwriter.writerow([labels[i], int(all_predictions[i]), sents[i]])
        print('Predictions saved to {}'.format(preds_output_file_path))

    # Save positive samples.
    positive_samples_file_path = preds_output_file_path.with_suffix('.positive')
    with open(positive_samples_file_path, 'w', encoding='utf-8', newline='') as f:
        for i in range(len(all_predictions)):
            if all_predictions[i] == 1:
                f.write(sents[i])
                f.write('\n')
        print('Positive samples saved to {}'.format(positive_samples_file_path))

    return positive_samples_file_path

default_batch_size = 64

def default_predict(input_file_path):
  run_dir = Path(__file__).parent.parent / 'data/1p_col_3p_share_does'
  checkpoint_name = 'clf-700'
  batch_size = 1  # TODO: modify batch_iter to use default_batch_size
  return predict(run_dir, checkpoint_name, input_file_path, batch_size)


def main():
    # Show warnings and errors only
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # File paths
    tf.flags.DEFINE_string('test_data_file', None, 'Input file path (a sent-per-line file).')
    tf.flags.DEFINE_string('run_dir', None, 'Restore the model from this run.')
    tf.flags.DEFINE_string('checkpoint', None, 'Restore the graph from this checkpoint.')
    # Test batch size
    tf.flags.DEFINE_integer('batch_size', default_batch_size, 'Test batch size')

    FLAGS = tf.app.flags.FLAGS
    run_dir = FLAGS.run_dir
    checkpoint_name = FLAGS.checkpoint
    test_data_file_path = FLAGS.test_data_file
    input_batch_size = FLAGS.batch_size

    preds_file_path = predict(run_dir, checkpoint_name, test_data_file_path, input_batch_size)
    print('Prediction file path: {}'.format(preds_file_path))
    


if __name__ == '__main__':
    main()
