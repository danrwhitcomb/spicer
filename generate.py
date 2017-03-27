import os, sys
import csv
import argparse
import random
import string

import numpy as np
import keras
import keras.backend as K

from util import setup_logging, get_binarizer

OUTPUT_LENGTH = 144
TRIM_CHAR = '.'

logger = None

'''
Argument parsing
'''
def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate text based on a Keras model')

    #Runtime related
    parser.add_argument('model_path', type=str,
                        help='Path to model to generate text from')
    parser.add_argument('--char-file', type=str, required=True,
                        help='File with characters sepearated by commas')

    parser.add_argument('--log-file', required=False, type=str, dest='log_file',
                        help='File path to log to. Logs to STDOUT if unspecified.')
    parser.add_argument('--output-path', required=False, type=str, dest='output_path',
                        help='File to output generated text to. If unspecified, text is printed to STDOUT')

    #Output related
    parser.add_argument('--seed', required=False, type=str, dest='seed',
                        default=random.choice(string.letters),
                        help='A seed string to start generation with. If unspecified, a random character will be chosen')
    parser.add_argument('-l', '--length', required=False, type=int, dest='length',
                        default=OUTPUT_LENGTH,
                        help='Desired output length. Default: %d' % OUTPUT_LENGTH)
    parser.add_argument('--trim', required=False, type=bool, dest='trim',
                        default=False,
                        help='If provided, output is trimmed to the last generated period')

    return parser.parse_args()


'''
Data
'''
#Load mappings from a CSV file with format:
# x,y
def load_list(path):
    if not os.path.isfile(path):
        log.error('Unable to find file at %s' % path)
        exit(1)

    with open(path, 'rb') as f:
        reader = csv.reader(f)
        return reader.next()

#Converts a string into a matrix of one-hot
# vectors where each row represents a character
# If string is not as long a input shape, remaining
# rows are zero
def build_seed_input(seed, input_shape, binarizer):
    seed_one_hot = binarizer.transform(list(seed))
    seed_input = np.zeros(shape=input_shape)

    seed_input[-len(seed_one_hot):] = seed_one_hot

    return seed_input

def write_result(path, result):
    if path:
        with open(path, 'w') as f:
            f.write(result)
    else:
        sys.stdout.write(result)
'''
Model
'''
#Loads a keras model from the given file
def load_model(path):
    if not os.path.isfile(path):
        log.error('Unable to find model at %s' % path)
        exit(1)

    return keras.models.load_model(path)

#Converts a model output to binary classification
# Max val in vector gets set to 1, all others are zero
def translate_model_output(output):
    output_one_hot = np.zeros(output.shape)
    for i, vect in enumerate(output):
        char_one_hot = np.zeros((len(vect)))
        char_one_hot[np.argmax(vect)] = 1

        output_one_hot[i] = char_one_hot

    return output_one_hot

#Given a mode, a starting input, a desired length
# and a LabelBinarizer, generate text
def generate_text(model, start, length, binarizer):

    current = start
    result = []

    #Right now, just produce `length` characters
    for i in range(length):
        output = model.predict(np.reshape(current, (1, current.shape[0], current.shape[1])))

        #Slide the input back one index, and append
        # the latest output to the end
        output_one_hot = translate_model_output(output[0])
        current[:-1] = current[1:]
        current[-1] = output_one_hot[-1]

        #Add new char to result
        result.append(output_one_hot[-1])

    #Convert the result to a string
    char_output = binarizer.inverse_transform(np.asarray(result))
    return ''.join(char_output.tolist())

#Cleans up generated text into something that
# is hopefully a syntactically valid sentence
def clean_text(raw_text, length, trim):

    #Get the last 'length' characters
    result = raw_text[-length:]

    #If 'trim' is on, get rid pf everything after the last period
    if trim:
        split_text = result.split('.')
        result = ''.join(split_text[:-1]) if len(split_text) > 1 else result

    return result



'''
Main
'''
def main():
    args = parse_arguments()
    logger = setup_logging(args.log_file)

    #Load int to char map
    logger.info('Loading characters')
    characters = load_list(args.char_file)

    #Load model
    logger.info('Loading model')
    model = load_model(args.model_path)

    #Make sure all the data is as anticipated
    seed = args.seed
    time_steps = model.input_shape[1]
    character_count = model.input_shape[2]

    if time_steps < len(seed):
        log.error('Seed is too large for model')
        exit(1)

    if len(characters) != character_count:
        logger.error('Character file size does not match model size')
        exit(1)

    #Convert seed to the proper network input shape
    logger.info('Building input from seed "%s"' % seed)
    binarizer = get_binarizer(characters)
    seed_input = build_seed_input(seed, (time_steps, character_count),
                                  binarizer)

    logger.info('Generating text')
    raw_text = generate_text(model, seed_input, args.length,
                                binarizer)

    cleaned_text = clean_text(raw_text, args.length, args.trim)
    write_output(clean_text)

if __name__ == '__main__':
    main()
