#!/env/bin/python
"""
COGS 118B Project
Author: Simon Fong, Thinh Le, Wilson Tran

"""

from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical
import numpy as np
import glob
import os
import cv2
import random
import matplotlib.pyplot as plt
from dataset import Dataset


IMAGE_WIDTH,IMAGE_HEIGHT,NUM_CHANNELS = 299,299,3
NUM_TRAIN,NUM_VAL,NUM_TEST = 1,1,98
EPOCHS = 1
BATCH_SIZE = 50

# Load dataset
cal = Dataset('caltech',IMAGE_HEIGHT,IMAGE_WIDTH)
cal.read_data()
cal.train_val_test_split(NUM_TRAIN,NUM_VAL,NUM_TEST)
num_classes = cal.num_classes

MEAN_PIXEL = np.array([104., 117., 123.]).reshape((1,1,3))


def load_model():
    base_model = InceptionV3(include_top=False, weights='imagenet',input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS))

    print('Model weights loaded.')
    base_out = base_model.output

    x = Flatten()(base_out)
    x = Dense(256,activation='relu')(x)
    x = Dropout(0.5)(x)

    predictions = Dense(num_classes,activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    print 'Build model'
    model.summary()

    model.compile(optimizers.SGD(lr=1e-4,momentum=0.9),'categorical_crossentropy', metrics=['accuracy'])

    print 'Compile model'
    return model




def main():
    # make model
    model = load_model()
    print 'Inception created\n'

    num_steps = int(cal.image_count / BATCH_SIZE/16)

    # Store data to plot
    train_acc = np.array([])
    train_val_acc = np.array([])
    train_loss = np.array([])
    train_val_loss = np.array([])
    """
    print "NUM STEPS = {}".format(num_steps)

    for i in range(num_steps):
        print('STEP {}------------------------------------------'.format(i))
        # read train and validation data and train the model for n epochs
        print 'Load train data: step {}'.format(i)
        X_train, Y_train = cal.next_batch(BATCH_SIZE)

        print 'Load val data: step {}'.format(i)
        X_val, Y_val = cal.next_batch(BATCH_SIZE)

        # TODO: Train model
        history = model.fit(x=X_train,y=Y_train,batch_size=BATCH_SIZE,epochs=EPOCHS,validation_data=(X_val,Y_val))

        train_acc = np.append(train_acc, history.history['acc'])
        train_val_acc = np.append(train_val_acc, history.history['val_acc'])
        train_loss = np.append(train_loss, history.history['loss'])
        train_val_loss = np.append(train_val_loss, history.history['val_loss'])

    """
    
    
    
    
    X_train, Y_train = cal.load_training()
    
    X_val, Y_val = cal.load_validation()
    
    history = model.fit(x=X_train,y=Y_train,batch_size=BATCH_SIZE,
                        epochs=EPOCHS,validation_data=(X_val,Y_val))

    train_acc = np.append(train_acc, history.history['acc'])
    train_val_acc = np.append(train_val_acc, history.history['val_acc'])
    train_loss = np.append(train_loss, history.history['loss'])
    train_val_loss = np.append(train_val_loss, history.history['val_loss'])
    
     
    # Save model weights
    model.save('fully_trained_1_1_98.h5')
    print 'model weights saved.'

    """
    # Create plots
    plt.figure()
    plt.hold(True)
    plt.plot(train_acc)
    plt.plot(train_val_acc)
    plt.legend(loc='upper right')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig('./acc_vs_val_acc.png')
    plt.hold(False)
    plt.show()

    plt.figure()
    plt.hold(True)
    plt.plot(train_loss)
    plt.plot(train_val_loss)
    plt.legend(loc='upper right')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('./loss_vs_val_loss.png')
    plt.hold(False)
    plt.show()
    """
    X_test, Y_test = cal.load_test()
    metrics = model.evaluate(x=X_test,y=Y_test, batch_size=BATCH_SIZE)
    
    print(metrics)
    print(model.metrics_names)

    return


if __name__ == '__main__':
    main()
