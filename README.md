# Introduction
The beginnings of a bot that uses LSTM-based networks to generate tweets. Based on Keras and scikit-learn.

## Data
Though the network is generally data agnostic, this specific data checked into the repo is cleaned transcripts of Sean Spicer White House press briefings. These were chosen because the briefings are available for free, it is mostly one person talking about one thing (making for better and hopefully funnier results), and I wanted to go through the exercise of pulling text data, cleaning it up, and running it through a custom model.

`net.py` will load all of the files in whatever folder is provided as the first CLI argument and parse them for text. Right now, this has all been setup with the Spicer data, but can be easily be replaced with another dataset.

## Dependencies
See `requirements.txt` for dependency information. Everything should be accessible via pip. There are no further requirements to train the model.

### Theano vs Tensorflow
The current requirements include both Tensorflow and Theano as requirements, but either can be used and you can configure Keras for your choice. I usually choose Theano because the GPU I run models on is too old for Tensorflow.

To choose, make a file at `~/.keras/keras.json` and make it look something like below. Switch out `theano` for `tensorflow` if you want Keras to load Tensorflow instead.

```
{
    "image_dim_ordering": "tf",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "theano"
}
```

## How to run
#### Training the network
You can start training the model with `net.py`. All that is required to begin training is the directory path which contains the data you want to load.

```
$ python net.py [data directory]
```

The script will always output two files. One is the trained Keras model ( `text-generation.model` by default), and the other is a `.csv` file which defines the characters the model was trained with (`chars.csv` by default). Both files are required by the generation script to produce new text.

You can specify output file paths as well customize network parameters by using additional flags. Do `python net.py --help` for more information.

#### Generating text
In order to generate new text with the products of `net.py`, you will need to run `generate.py`.

`generate.py` requires a path to the model and a path to the character CSV file that were produced by  `net.py`.

```
$ python generate.py [model path] --char-file [char csv]
```

Text will be sent to STDOUT by default. You can use the `--output-path` flag to write it to a specific file. Additional options are available via `python generate.py --help`.
