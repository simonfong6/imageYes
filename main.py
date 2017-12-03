from inception_v3 import inception_v3
from dataset import Dataset
import tensorflow as tf
import cv2
import numpy as np

IMAGE_WIDTH = 299
IMAGE_HEIGHT = 299
EPOCHS = 100
BATCH_SIZE = 50

# Load dataset
cal = Dataset('caltech',IMAGE_HEIGHT,IMAGE_WIDTH)
cal.read_data()
num_classes = cal.num_classes

# Placeholder inputs and output
inputs = tf.placeholder(tf.float32, [None,IMAGE_HEIGHT,IMAGE_WIDTH,3])
predict = tf.placeholder(tf.float32, [None,num_classes])

y_conv, some = inception_v3(inputs, num_classes=num_classes)

# Cross entropy graph
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=predict, logits=y_conv))

# Training graph
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Predictions graph
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(predict, 1))

# Accuracy graph
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    for i in range(EPOCHS):
        images, labels = cal.next_batch(BATCH_SIZE)
        
        if i % 10 == 0:
            print(labels[0].shape)
            train_accuracy = accuracy.eval(feed_dict={
                inputs: images, predict: labels})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        
        _, loss = sess.run([train_step, cross_entropy], 
                            feed_dict={inputs: images, predict: labels})
        
