from inception_v3 import inception_v3
from data import Data
import tensorflow as tf
import cv2
import numpy as np

IMAGE_WIDTH = 299
IMAGE_HEIGHT = 299
NUM_CLASSES = 257

caltech_256 = Data('256_ObjectCategories',IMAGE_HEIGHT,IMAGE_WIDTH)

caltech_256.load_data()

images = caltech_256.images
labels = caltech_256.labels


inputs = tf.placeholder("float", [None,IMAGE_HEIGHT,IMAGE_WIDTH,3])
y_ = tf.placeholder("float", [None,1,1,NUM_CLASSES])

y = np.zeros((1,1,NUM_CLASSES))

base, some = inception_v3(inputs, num_classes=NUM_CLASSES)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    poop = base.eval(feed_dict={inputs: images, y_: labels})
    print(poop)
