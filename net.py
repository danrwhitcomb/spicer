'''
Simple text generation neural network
'''
import os, sys
import csv
import argparse

import numpy as np

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import BaseLogger, EarlyStopping


from util import setup_logging, get_binarizer

CHAR_CSV = 'chars.csv'
MODEL_NAME = 'text-generation.model'

TIME_LENGTH = 75
STEP_LENGTH = 5
DROPOUT = 0.5
BATCH_SIZE = 32
EPOCHS = 100

logger = None

'''
Argument parsing
'''
def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a text generation model')

    #Management arguments
    parser.add_argument('data_dir', type=str, help='Directory which holds training data')
    parser.add_argument('--model-name', required=False, type=str, dest='model_name',
                        default=MODEL_NAME, help='Output location for the trained model (Default: "%s")' % MODEL_NAME)
    parser.add_argument('--character-map', required=False, type=str, dest='character_map',
                        default=CHAR_CSV, help='Path to save int to char mappings (Default: "%s")' % CHAR_CSV)
    parser.add_argument('--log-file', required=False, type=str, dest='log_file',
                        help='File path to log to. Logs to STDOUT if unspecified.')

    #Network parameters
    parser.add_argument('--batch-size', type=int, dest='batch_size', required=False,
                        default=BATCH_SIZE, help='Batch size to train on (Default: %d)' % BATCH_SIZE)
    parser.add_argument('--epochs', type=int, dest='epochs', required=False,
                        default=EPOCHS, help='Number of training epochs (Default: %d)' % EPOCHS)
    parser.add_argument('--dropout', type=int, dest='dropout', required=False,
                        default=DROPOUT, help='Dropout percentage range between (0, 1). (Default: %.2f)' % DROPOUT)

    return parser.parse_args()


'''
Data Management
'''
# Loads raw text from all the files
# in a directory
def load_text(data_dir):
    if not os.path.isdir(data_dir):
        print("%s is not a directory" % data_dir)
        exit(1)

    data = os.listdir(data_dir)
    raw_text = ''
    for filename in data:
        path = os.path.join(data_dir, filename)
        with open(path, 'r') as f:
            raw_text += f.read()

    return raw_text

#Writes list to a file separated by commas
def save_list(path, lst):
    with open(path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(lst)

#Gets all of the unique characters in a string
def get_chars(raw_text):
    return sorted(list(set(raw_text)))

#Translates a batch of string to
# one-hot vectors by character given a LabelBinarizer
def convert_to_one_hot_batch(batch, binarizer):
    one_hot_batch = []
    for item in batch:
        one_hot_batch.append(binarizer.transform(list(item)))

    return np.array(one_hot_batch)

# Parse raw_text into example
# and label lists
def create_examples(raw_text):
    examples = []
    labels = []
    for i in range(0, len(raw_text) - TIME_LENGTH + 1, STEP_LENGTH):
        start = i
        end = i + TIME_LENGTH
        examples.append(raw_text[start:end])
        labels.append(raw_text[start + 1: end + 1])

    return examples, labels

'''
Network setup
'''
def get_new_network(input_shape, dropout):
    logger.info('Creating model with input shape: %s' % str(input_shape))
    model = Sequential()
    model.add(LSTM(100, activation='tanh', input_shape=input_shape,
        dropout=dropout, return_sequences=True))
    model.add(LSTM(200, dropout=dropout, return_sequences=True))
    model.add(TimeDistributed(Dense(input_shape[1])))
    model.add(Activation('softmax'))

    optimizer = get_optimizer()
    loss = get_loss()

    logger.info('Compiling model')
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def get_optimizer():
    return Adam()

def get_loss():
    return 'mean_squared_error'

'''
MAIN
'''
def main():
    global logger

    args = parse_arguments()
    logger = setup_logging(args.log_file)

    #Get opts and load data
    logger.info('Loading raw text')
    data_dir = args.data_dir
    raw_text = load_text(data_dir)

    #Save the character mappings to be used
    # for character regeneration
    logger.info('Saving character map')
    characters = get_chars(raw_text)
    save_list(args.character_map, characters)

    #Create examples from rax data
    logger.info('Creating examples')
    char_examples, char_labels = create_examples(raw_text)

    #Convert char vectors to matrices where each row
    # is a one-hot vector
    logger.info('Converting to one-hot representations')
    binarizer = get_binarizer(characters)
    one_hot_examples = convert_to_one_hot_batch(char_examples, binarizer)
    one_hot_labels = convert_to_one_hot_batch(char_labels, binarizer)

    #Get the model and fit
    logger.info('Building model')
    model = get_new_network(input_shape=(one_hot_examples.shape[1], one_hot_examples.shape[2]), dropout=args.dropout)

    logger.info('Fitting model')
    model.fit(one_hot_examples, one_hot_labels, batch_size=args.batch_size, epochs=args.epochs)

    logger.info('Saving model to %s' % args.model_name)
    model.save(args.model_name)

if __name__ == '__main__':
    main()
