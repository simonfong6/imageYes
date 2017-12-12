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
import json                                     # Writing data to logger
from dataset import Dataset                     # Custom Dataset

DATASET_NAME = 'caltech'
IMAGE_WIDTH,IMAGE_HEIGHT,NUM_CHANNELS = 299,299,3

ID = str(sys.argv[1])

# Load dataset
cal = Dataset(DATASET_NAME,IMAGE_HEIGHT,IMAGE_WIDTH)
cal.read_data()
names = cal.names


# Video capturing inits
cap = cv2.VideoCapture(0) 

def logger(message):
    """Logs any message into a file"""
    with open('./models/stats.txt', 'a+') as f:
        print >>f, message
        print(message)
        
def take_picture():
    """Takes a picture and returns it in the proper format"""
    
    # Capture frame-by-frame
    ret, image = cap.read()

    #
    images = []

    # Resize image
    image_resized = cv2.resize(image, (IMAGE_HEIGHT,IMAGE_WIDTH))
    images.append(image_resized)
    
    images = np.array(images)

    return images


def main():
    
    # Make model
    model = load_model(ID)
    print("Model created\n")

    while(True):
        image = take_picture()
        
        label = model.predict(image)
        
        label = names[np.argmax(label[0])]
        #label = 'foo'
        print(label)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        image = cv2.resize(image[0], (2*IMAGE_HEIGHT,2*IMAGE_WIDTH))
        cv2.putText(image,label,(50,50), font, 1,(255,0,0),2,cv2.LINE_AA)
        cv2.imshow('image',image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    return 0


if __name__ == '__main__':
    main()
