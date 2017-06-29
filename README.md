# Multi-class Text Classification
Implement three neural networks in Tensorflow for multi-class text classification problem.
## Models
* A LSTM classifier.
* A CNN classifier. Reference: [Implementing a CNN for Text Classification in Tensorflow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/).
* A C-LSTM classifier. Reference: [A C-LSTM Neural Network for Text Classification](https://arxiv.org/abs/1511.08630).
## Train
Run train.py to train the models.
Parameters:
'''
  --clf CLF             Type of classifiers to use. You have three choices:
                        ['cnn', 'rnn', 'clstm]
  --data_file DATA_FILE
                        Data file path
  --stop_word_file STOP_WORD_FILE
                        Stop word file path
  --language LANGUAGE   Language of the data file. You have two choices:
                        ['ch', 'en']
  --min_frequency MIN_FREQUENCY
                        Minimal word frequency
  --num_classes NUM_CLASSES
                        Number of classes
  --max_length MAX_LENGTH
                        Length the longest sentence in the document
  --vocab_size VOCAB_SIZE
                        Vocabulary size
  --test_size TEST_SIZE
                        Test size
  --embedding_size EMBEDDING_SIZE
                        Word embedding size
  --filter_sizes FILTER_SIZES
                        CNN filter size
  --num_filters NUM_FILTERS
                        Number of filters per filter size
  --hidden_size HIDDEN_SIZE
                        Number of hidden units in the LSTM cell
  --num_layers NUM_LAYERS
                        Number of the LSTM cells
  --keep_prob KEEP_PROB
                        Dropout keep probability
  --learning_rate LEARNING_RATE
                        Learning rate
  --l2_reg_lambda L2_REG_LAMBDA
                        L2 regularization lambda
  --batch_size BATCH_SIZE
                        Batch size
  --num_epochs NUM_EPOCHS
                        Number of epochs
  --evaluate_every_steps EVALUATE_EVERY_STEPS
                        Evaluate the model on validation set after this many
                        steps
  --save_every_steps SAVE_EVERY_STEPS
                        Save the model after this many steps
  --num_checkpoint NUM_CHECKPOINT
                        Number of models to store
'''

