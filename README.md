# Multi-class Text Classification
Implement three neural networks in Tensorflow for multi-class text classification problem.
## Models
* A LSTM classifier. See rnn_classifier.py
* A Bidirectional LSTM classifier. See rnn_clf.py
* A CNN classifier. See cnn_classifier.py. Reference: [Implementing a CNN for Text Classification in Tensorflow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/).
* A C-LSTM classifier. See clstm_classifier.py. Reference: [A C-LSTM Neural Network for Text Classification](https://arxiv.org/abs/1511.08630).
## Data Format
Training data should be stored in csv file. The first line of the file should be ["label", "content"] or ["content", "label"].
## Train
Run train.py to train the models.
Parameters:
```
python train.py --help
```
```
optional arguments:
  -h, --help            show this help message and exit
  --clf CLF             Type of classifiers. Default: cnn. You have four
                        choices: [cnn, lstm, blstm, clstm]
  --data_file DATA_FILE
                        Data file path
  --stop_word_file STOP_WORD_FILE
                        Stop word file path
  --language LANGUAGE   Language of the data file. You have two choices: [ch,
                        en]
  --min_frequency MIN_FREQUENCY
                        Minimal word frequency
  --num_classes NUM_CLASSES
                        Number of classes
  --max_length MAX_LENGTH
                        Max document length
  --vocab_size VOCAB_SIZE
                        Vocabulary size
  --test_size TEST_SIZE
                        Cross validation test size
  --embedding_size EMBEDDING_SIZE
                        Word embedding size. For CNN, C-LSTM.
  --filter_sizes FILTER_SIZES
                        CNN filter sizes. For CNN, C-LSTM.
  --num_filters NUM_FILTERS
                        Number of filters per filter size. For CNN, C-LSTM.
  --hidden_size HIDDEN_SIZE
                        Number of hidden units in the LSTM cell. For LSTM, Bi-
                        LSTM
  --num_layers NUM_LAYERS
                        Number of the LSTM cells. For LSTM, Bi-LSTM, C-LSTM
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
```
You could run train.py to start training. For example:
```
python train.py --data_file=./data/data.csv --clf=lstm
```
