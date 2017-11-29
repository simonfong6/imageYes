from inception_v3 import inception_v3
import tensorflow as tf
import cv2
import numpy as np

IMAGE_WIDTH = 299
IMAGE_HEIGHT = 299
NUM_CLASSES = 256

image = cv2.imread('256_ObjectCategories/001.ak47/001_0001.jpg')

small = cv2.resize(image, (IMAGE_HEIGHT,IMAGE_WIDTH))





inputs = tf.placeholder("float", [None,IMAGE_HEIGHT,IMAGE_WIDTH,3])
y_ = tf.placeholder("float", [None,1,1,NUM_CLASSES])

y = np.zeros((1,1,NUM_CLASSES))

base, some = inception_v3(inputs, num_classes=NUM_CLASSES)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    poop = base.eval(feed_dict={inputs: [small], y_: [y]})
    print poop
