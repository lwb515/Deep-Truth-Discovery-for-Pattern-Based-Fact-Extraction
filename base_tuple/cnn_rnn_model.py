import tensorflow as tf
import utils.data_loader as dl
import tensorflow.contrib.rnn as rnn


class Model:
    def __init__(self, data_file_path, model_para):
        max_length, vec_dim, cnn_hidden_size, cnn_kernel_size, cnn_keep_prob, rnn_num_units, rnn_input_keep_prob, \
            rnn_output_keep_prob, test_words, rnn_type = model_para

        # load data
        self.patterns, [self.x_e, self.x_a, self.x_v], self.pat_freq, self.y_info, self.freq_tot, self.pat_infos, \
            self.pat_labels = dl.get_pattern_info_vec(data_file_path, vec_dim, False, max_length, test_words)

        self.max_length = max_length

        # placeholder
        self.y = tf.placeholder(name="y", shape=[None, 1], dtype=tf.int64)
        self.e = tf.placeholder(name="e", shape=[None, max_length, vec_dim], dtype=tf.float32)
        self.a = tf.placeholder(name="a", shape=[None, max_length, vec_dim], dtype=tf.float32)
        self.v = tf.placeholder(name="v", shape=[None, max_length, vec_dim], dtype=tf.float32)
        self.f = tf.placeholder(name="freq", shape=[None, 1], dtype=tf.float32)
        self.f_tot = tf.placeholder(name="freq_tot", dtype=tf.int64)

        # CNN
        e_f = self.__cnn(x=self.e, keep_prob=cnn_keep_prob, hidden_units=cnn_hidden_size,
                         kernel_size=cnn_kernel_size)
        a_f = self.__cnn(x=self.v, keep_prob=cnn_keep_prob, hidden_units=cnn_hidden_size,
                         kernel_size=cnn_kernel_size)
        v_f = self.__cnn(x=self.a, keep_prob=cnn_keep_prob, hidden_units=cnn_hidden_size,
                         kernel_size=cnn_kernel_size)
        self.feature = tf.reshape(tf.concat([e_f, a_f, v_f], axis=1), shape=[-1, 3, cnn_hidden_size])

        batch_size = tf.shape(self.feature)[0]

        # network
        state = self.rnn(x=self.feature, model=rnn_type, batch_size=tf.shape(self.feature)[0],
                         num_units=rnn_num_units, input_keep_prob=rnn_input_keep_prob,
                         output_keep_prob=rnn_output_keep_prob)

        y_pre = self.__ann_classifier(state, rnn_num_units)
        self.tuple_score = tf.split(y_pre, 2, 1)[1]
        self.loss = tf.reduce_mean(
            tf.multiply(self.f, tf.square(tf.cast(self.y, dtype=tf.float32) - self.tuple_score)))

        # prediction
        self.prediction_result = tf.argmax(y_pre, axis=1, name="prediction_result")
        correct_prediction = tf.equal(self.y, tf.reshape(self.prediction_result, [batch_size, 1]))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32), name="accuracy")

    def __cnn(self, x, hidden_units=128, kernel_size=1, keep_prob=1.0):
        with tf.variable_scope('cnn_model', reuse=tf.AUTO_REUSE):
            x = tf.layers.conv1d(inputs=x,
                                 filters=hidden_units,
                                 kernel_size=kernel_size,
                                 kernel_initializer=tf.contrib.layers.xavier_initializer())

            x = tf.reduce_max(x, axis=-2)

            x = tf.nn.relu(x)
            x = tf.contrib.layers.dropout(x, keep_prob=keep_prob)

            return x

    def __ann_classifier(self, state, hidden_size):
        with tf.variable_scope('Ann__discover', reuse=tf.AUTO_REUSE):
            W = tf.get_variable(name="classifier_W",
                                initializer=tf.truncated_normal([hidden_size, 2], stddev=0.1, name="W"),
                                dtype=tf.float32)

            return tf.cast(tf.nn.softmax(tf.matmul(state, W)), dtype=tf.float32)

    def rnn(self, x, model="lstm", batch_size=128, num_units=100, input_keep_prob=1.0, output_keep_prob=1.0):
        with tf.variable_scope('lstm_and_gru_model', reuse=tf.AUTO_REUSE):
            if model == "lstm":
                rnn_cell = tf.nn.rnn_cell.LSTMCell(name='basic_lstm_cell', num_units=num_units)
            else:
                rnn_cell = rnn.GRUCell(num_units=num_units)

            rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, input_keep_prob=input_keep_prob,
                                                     output_keep_prob=output_keep_prob)

            init_state = rnn_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
            _, states = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=x, initial_state=init_state, scope="dynamic-rnn")
            if model == "lstm":
                return states[1]
            else:
                return states

