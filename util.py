import os, sys
import logging
import numpy as np
from sklearn.preprocessing import LabelBinarizer

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

#Gets a fitted LabelBinarizer
def get_binarizer(classes):
    binarizer = LabelBinarizer()
    binarizer.fit(classes)
    return binarizer
