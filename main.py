from inception_v3 import inception_v3_base
import tensorflow as tf


inputs = tf.placeholder("float", [10,100,100,3])
base = inception_v3_base(inputs)
