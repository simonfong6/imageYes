from inception_v3 import inception_v3
import tensorflow as tf
import cv2
import numpy as np

image = cv2.imread('256_ObjectCategories/001.ak47/001_0001.jpg')

small = cv2.resize(image, (299,299))





inputs = tf.placeholder("float", [None,299,299,3])
y_ = tf.placeholder("float", [None,1,1,256])

y = np.zeros((1,1,256))

base, some = inception_v3(inputs, num_classes=256)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    poop = base.eval(feed_dict={inputs: [small], y_: [y]})
    print poop
