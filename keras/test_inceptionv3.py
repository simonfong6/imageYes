#!/env/bin/python
"""
COGS 118B Project
Author: Simon Fong, Thinh Le, Wilson Tran
"""

from keras.models import Model, load_model
import numpy as np
import os
import sys
import cv2
import random
import matplotlib
import json                                     # Writing data to logger
matplotlib.use('Agg')                           # Stops from plotting to screen
import matplotlib.pyplot as plt
from dataset import Dataset                     # Custom Dataset

DATASET_NAME = 'caltech'
IMAGE_WIDTH,IMAGE_HEIGHT,NUM_CHANNELS = 299,299,3
NUM_TRAIN,NUM_VAL,NUM_TEST = 40,5,55
BATCH_SIZE = 50

ID = str(sys.argv[1])

# Load dataset
cal = Dataset(DATASET_NAME,IMAGE_HEIGHT,IMAGE_WIDTH)
cal.read_data()
cal.train_val_test_split(NUM_TRAIN,NUM_VAL,NUM_TEST)
num_classes = cal.num_classes

def logger(message):
    """Logs any message into a file"""
    with open('./models/stats.txt', 'a+') as f:
        print >>f, message
        print(message)


def main():
    
    # Make model
    model = load_model(ID)
    print("Model created\n")

    
    # Test the model
    X_test, Y_test = cal.load_testing()
    metrics = model.evaluate(x=X_test,y=Y_test, batch_size=BATCH_SIZE)
    
    logger(ID)
    logger(metrics)
    logger(model.metrics_names)

    return 0


if __name__ == '__main__':
    main()
