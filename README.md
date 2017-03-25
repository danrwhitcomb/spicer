##Introduction
The beginnings of a bot that uses LSTM-based networks to generate tweets. Based on Keras and scikit-learn.

##Data
Though the network is generally data agnostic, this specific data checked into the repo is cleaned transcripts of Sean Spicer White House press briefings. These were chosen because the briefings are available for free, it is mostly one person talking about one thing (making for better and hopefully funnier results), and I wanted to go through the exercise of pulling text data, cleaning it up, and running it through a custom model.

`net.py` will load all of the files in whatever folder is provided as the first CLI argument and parse them for text. Right now, this has all been setup with the Spicer data, but can be easily be replaced with another dataset.

##Dependencies
See `requirements.txt` for dependency information. Everything should be accessible via pip. There are no further requirements to train the model.

###Theano vs Tensorflow
The current requirements include both Tensorflow and Theano as requirements, but either can be used and you can configure Keras for your choice. I usually choose Theano because the GPU I run models on is too old for tensorflow.

To choose, make a file at `~/.keras/keras.json` and make it look something like below. Switch out `theano` for `tensorflow` if you want to use something different.

```
{
    "image_dim_ordering": "tf",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "theano"
}
```

##How to run
You can start training the model with `net.py`. You will need to provide the data directory you want to load, as well as the directory you want for the character-integer mappings.

```
$ python net.py [data directory] [mappings output directory]
```

For example:
```
$ python net.py ~/danrwhitcomb/spicer/data ~/danrwhitcomb/spicer
```
