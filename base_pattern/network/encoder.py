import tensorflow as tf
import tensorflow.contrib.rnn as tfrnn
import numpy as np


def __dropout__(x, keep_prob=1.0):
    return tf.contrib.layers.dropout(x, keep_prob=keep_prob)


def __pooling__(x):
    return tf.reduce_max(x, axis=-2)


def __cnn_cell__(x, hidden_size=100, kernel_size=1, stride_size=1):
    x = tf.layers.conv1d(inputs=x,
                         filters=hidden_size,
                         kernel_size=kernel_size,
                         strides=stride_size,
                         padding='same',
                         kernel_initializer=tf.contrib.layers.xavier_initializer())
    return x


def cnn(x, hidden_size=100, kernel_size=1, stride_size=1, activation=tf.nn.relu, var_scope=None, keep_prob=1.0):
    with tf.variable_scope(var_scope or "cnn", reuse=tf.AUTO_REUSE):
        x = __cnn_cell__(x, hidden_size, kernel_size, stride_size)
        x = __pooling__(x)
        x = activation(x)
        x = __dropout__(x, keep_prob)
        return x


def __lstm_cell__(num_units):
    return tf.nn.rnn_cell.LSTMCell(name='basic_lstm_cell', num_units=num_units)


def __gru_cell__(num_units):
    return tfrnn.GRUCell(num_units=num_units)


def rnn(x, model="lstm", batch_size=128, num_units=100, input_keep_prob=1.0, output_keep_prob=1.0):
    with tf.variable_scope('lstm_and_gru_model', reuse=tf.AUTO_REUSE):
            if model == "lstm":
                rnn_cell = __lstm_cell__(num_units=num_units)
            else:
                rnn_cell = tfrnn.GRUCell(num_units=num_units)

            rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, input_keep_prob=input_keep_prob,
                                                     output_keep_prob=output_keep_prob)

            init_state = rnn_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
            _, states = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=x, initial_state=init_state, scope="dynamic-rnn")
            if model == "lstm":
                return states[1]
            else:
                return states


def rnn_score(x, batch_size=128, num_units=100):
    with tf.variable_scope('rnn_score', reuse=tf.AUTO_REUSE):
        W = tf.get_variable(name="rnn_score_weight",
                            initializer=tf.truncated_normal([num_units, 2], stddev=0.1), dtype=tf.float32)
        bias = tf.get_variable(name="rnn_score_bias",
                               initializer=tf.constant(0.1, shape=[batch_size, 2]), dtype=tf.float32)

        return tf.cast(tf.nn.softmax(tf.matmul(x, W) + bias), dtype=tf.float32)
        # return tf.split(tf.cast(tf.nn.softmax(tf.matmul(x, W) + bias), dtype=tf.float32), 2, 1)[1]
