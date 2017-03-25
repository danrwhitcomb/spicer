'''
Simple text generation neural network
'''
import os, sys
import csv
import logging
import argparse

import numpy as np

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
from sklearn.preprocessing import LabelBinarizer

INT_TO_CHAR_CSV = 'int-to-char.csv'
CHAR_TO_INT_CSV = 'char-to-int.csv'
MODEL_NAME = 'text-generation.model'

TIME_LENGTH = 75
STEP_LENGTH = 5
DROPOUT = 0.5
BATCH_SIZE = 32
EPOCHS = 100

logger = None

'''
General Utilities
'''
def setup_logging(log_file):
    logger = logging.getLogger()
    ch = None
    if log_file is not None:
        ch = logging.FileHandler(log_file)
    else:
        ch = logging.StreamHandler(sys.stdout)

    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)
    return logger

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a text generation model')

    #Management arguments
    parser.add_argument('data_dir', type=str, help='Directory which holds training data')
    parser.add_argument('--model-name', required=False, type=str, dest='model_name',
                        default=MODEL_NAME, help='Output location for the trained model (Default: "%s")' % MODEL_NAME)
    parser.add_argument('--itoa-map', required=False, type=str, dest='itoa_map',
                        default=INT_TO_CHAR_CSV, help='Path to save int to char mappings (Default: "%s")' % INT_TO_CHAR_CSV)
    parser.add_argument('--atoi-map', required=False, type=str, dest='atoi_map',
                        default=CHAR_TO_INT_CSV, help='Path to save char to int mappings (Default: "%s")' % CHAR_TO_INT_CSV)
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

#Writes a dictionary to a file
def save_char_mapping(path, d):
    with open(path, 'w') as f:
        w = csv.writer(f)
        w.writerows(d.items())

#Gets the unique characters in a string
# and maps them to integers.
def get_char_mappings(raw_text):
    characters = sorted(list(set(raw_text)))
    return ({c:i for i, c in enumerate(characters)},
           {i:c for i, c in enumerate(characters)})

#Gets a fitted LabelBinarizer
def get_binarizer(ints):
    binarizer = LabelBinarizer()
    binarizer.fit(ints)
    return binarizer

#Coverts a vector to another vector
# given the mappings
def translate_vector(vector, mappings):
    translate_vect = np.zeros(len(vector))
    for i, val in enumerate(vector):
        translate_vect[i] = mappings[val]

    return translate_vect

#Converts all vectors in a list based on
# the given mappings
def translate_batch(batch, mappings):
    translated_batch = []
    for item in batch:
        translated_batch.append(translate_vector(item, mappings))

    return np.array(translated_batch)

#Translates a batch of items to
# one-hot vectors given a LabelBinarizer
def convert_to_one_hot_batch(batch, binarizer):
    one_hot_batch = []
    for item in batch:
        one_hot_batch.append(binarizer.transform(item))

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
        labels.append(raw_text[end])

    return examples, labels

'''
Network setup
'''
def get_new_network(input_shape, dropout):
    logger.info('Creating model with input shape: %s' % str(input_shape))
    model = Sequential()
    model.add(LSTM(TIME_LENGTH, activation='tanh', input_shape=input_shape,
        dropout=dropout, return_sequences=True))
    model.add(LSTM(TIME_LENGTH * 2, dropout=dropout))
    model.add(Dense(input_shape[1]))
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
    logger.info('Saving character maps')
    char_to_int, int_to_char = get_char_mappings(raw_text)
    save_char_mapping(args.itoa_map, int_to_char)
    save_char_mapping(args.atoi_map, char_to_int)

    #Create examples from rax data
    logger.info('Creating examples')
    char_examples, char_labels = create_examples(raw_text)

    #Convert characters to ints
    logger.info('Converting to integer representations')
    int_examples = translate_batch(char_examples, char_to_int)
    int_labels = translate_vector(char_labels, char_to_int)

    #Convert int vectors to one-hot labels
    logger.info('Converting to one-hot representations')
    binarizer = get_binarizer(int_to_char.keys())
    one_hot_examples = convert_to_one_hot_batch(int_examples, binarizer)
    one_hot_labels = binarizer.transform(int_labels)

    #Get the model and fit
    logger.info('Building model')
    model = get_new_network(input_shape=(None, one_hot_examples.shape[2]), dropout=args.dropout)

    logger.info('Fitting model')
    model.fit(one_hot_examples, one_hot_labels, batch_size=args.batch_size, epochs=args.epochs)

    logger.info('Saving model to %s' % args.model_name)
    model.save(args.model_name)

if __name__ == '__main__':
    main()
