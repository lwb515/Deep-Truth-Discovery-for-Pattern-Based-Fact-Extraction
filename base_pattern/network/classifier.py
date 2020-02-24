import tensorflow as tf
import numpy as np


def pattern_classifier(x, shape):
    with tf.variable_scope('pattern_classifier', reuse=tf.AUTO_REUSE):
        W = tf.get_variable(name="pattern_classifier_weight",
                            initializer=tf.truncated_normal([shape[0], 2], stddev=0.1), dtype=tf.float32)
        bias = tf.get_variable(name="pattern_classifier_bias",
                               initializer=tf.constant(0.1, shape=[shape[1], 2]), dtype=tf.float32)

        return tf.cast(tf.nn.softmax(tf.matmul(x, W) + bias), dtype=tf.float32)
