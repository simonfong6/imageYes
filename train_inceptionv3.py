"""
COGS 118B Project
Author: Simon Fong, Thinh Le, Wilson Tran

"""

from tensorflow.keras.python.keras.models import Model
from tensorflow.keras.python.keras.applications.vgg16 import VGG16, InceptionV3
from tensorflow.keras.python.keras import optimizers
from tensorflow.keras.python.keras.layers import Dropout, Flatten, Dense
from tensorflow.keras.python.keras.utils.np_utils import to_categorical
import numpy as np
import glob
import os
import cv2
import random
from dataset import Dataset


IMAGE_WIDTH = 299
IMAGE_HEIGHT = 299
EPOCHS = 100
BATCH_SIZE = 50

# Load dataset
cal = Dataset('caltech',IMAGE_HEIGHT,IMAGE_WIDTH)
cal.read_data()
num_classes = cal.num_classes

IMG_H, IMG_W, NUM_CHANNELS = IMAGE_HEIGHT, IMAGE_WIDTH, 3
MEAN_PIXEL = np.array([104., 117., 123.]).reshape((1,1,3))



def load_model():
    # TODO: use VGG16 to load lower layers of vgg16 network and declare it as base_model
    # TODO: use 'imagenet' for weights, include_top=False, (IMG_H, IMG_W, NUM_CHANNELS) for input_shape
    base_model = InceptionV3(include_top=False, weights='imagenet',input_shape=(IMG_H, IMG_W, NUM_CHANNELS))

    print('Model weights loaded.')
    base_out = base_model.output
    # TODO: add a flatten layer, a dense layer with 256 units, a dropout layer with 0.5 rate,
    # TODO: and another dense layer for output. The final layer should have the same number of units as classes

    x = Flatten()(base_out)
    x = Dense(256,activation='relu')(x)
    x = Dropout(0.5)(x)    

    predictions = Dense(19,activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    print 'Build model'
    model.summary()

    # TODO: compile the model, use SGD(lr=1e-4,momentum=0.9) for optimizer, 'categorical_crossentropy' for loss,
    # TODO: and ['accuracy'] for metrics
    model.compile(optimizers.SGD(lr=1e-4,momentum=0.9),'categorical_crossentropy', metrics=['accuracy'])

    print 'Compile model'
    return model



def main():
    # make model
    model = load_model()
    print 'VGG16 created\n'

    # read train and validation data and train the model for n epochs
    print 'Load train data:'
    X_train, Y_train = load_data(TRAIN_DIR)
    print 'Load val data:'
    X_val, Y_val = load_data(VAL_DIR)
    # TODO: Train model
    model.fit(x=X_train,y=Y_train,batch_size=BATCH_SIZE,epochs=NUM_EPOCHS,validation_data=(X_val,Y_val))

    # TODO: Save model weights
    model.save('side_hoe_number_2.h5')
    print 'model weights saved.'
    return


if __name__ == '__main__':
    main()
