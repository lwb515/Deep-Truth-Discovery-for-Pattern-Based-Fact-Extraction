import tensorflow as tf
import utils.data_loader as dl
import base_pattern.network.encoder as encoder
import base_pattern.network.classifier as classifier


class Model:
    def __init__(self, data_file, model_para):
        max_length, vec_dim, cnn_hidden_size, cnn_kernel_size, cnn_stride_size, cnn_keep_prob, rnn_num_units, \
            rnn_input_keep_prob, rnn_output_keep_prob, balance, test_words, with_truth_weight, rnn_type = model_para
        self.patterns, [self.x_e, self.x_a, self.x_v], self.pat_freq, self.y_info, self.freq_tot, self.tuple_labels,\
            self.tuple_info = dl.get_pattern_info_vec2(data_file, vec_dim=vec_dim, max_length=max_length, flag=test_words)

        # placeholder
        self.y = tf.placeholder(name="y", dtype=tf.int64)
        self.e = tf.placeholder(name="e", shape=[None, max_length, vec_dim], dtype=tf.float32)
        self.a = tf.placeholder(name="a", shape=[None, max_length, vec_dim], dtype=tf.float32)
        self.v = tf.placeholder(name="v", shape=[None, max_length, vec_dim], dtype=tf.float32)
        self.f = tf.placeholder(name="freq", shape=[None, 1], dtype=tf.int64)
        self.f_tot = tf.placeholder(name="freq_tot", dtype=tf.int64)
        self.t_y = tf.placeholder(name="tuples_label", shape=[None, 1], dtype=tf.int64)

        # CNN
        e_f = encoder.cnn(x=self.e, keep_prob=cnn_keep_prob, hidden_size=cnn_hidden_size,
                          kernel_size=cnn_kernel_size, stride_size=cnn_stride_size)
        a_f = encoder.cnn(x=self.v, keep_prob=cnn_keep_prob, hidden_size=cnn_hidden_size,
                          kernel_size=cnn_kernel_size, stride_size=cnn_stride_size)
        v_f = encoder.cnn(x=self.a, keep_prob=cnn_keep_prob, hidden_size=cnn_hidden_size,
                          kernel_size=cnn_kernel_size, stride_size=cnn_stride_size)
        self.feature = tf.reshape(tf.concat([e_f, a_f, v_f], axis=1), shape=[-1, 3, cnn_hidden_size])

        # RNN
        state = encoder.rnn(x=self.feature, model=rnn_type, batch_size=tf.shape(self.feature)[0],
                            num_units=rnn_num_units, input_keep_prob=rnn_input_keep_prob,
                            output_keep_prob=rnn_output_keep_prob)

        # RNN_score ANN
        self.scores = encoder.rnn_score(x=state, batch_size=1, num_units=rnn_num_units)

        self.tuple_pred = tf.argmax(self.scores, axis=1, name="tuples_pred")
        tuple_corr_pred = tf.equal(self.t_y, tf.reshape(self.tuple_pred, [tf.shape(self.scores)[0], 1]))
        self.tuple_acc = tf.reduce_mean(tf.cast(tuple_corr_pred, dtype=tf.float32), name="tuple_acc")
        self.scores = tf.split(self.scores, 2, 1)[1]

        # pattern encode
        cnn_feature = tf.concat([e_f, a_f, v_f], axis=-1)
        if with_truth_weight:
            pattern_encode_middle = tf.reduce_sum(tf.multiply(tf.multiply(self.scores, tf.cast(self.f, dtype=tf.float32)),
                                                              cnn_feature) / tf.cast(self.f_tot, dtype=tf.float32), 0)
        else:
            pattern_encode_middle = tf.reduce_sum(tf.multiply(tf.cast(self.f, dtype=tf.float32),
                                                              cnn_feature) / tf.cast(self.f_tot, dtype=tf.float32), 0)
        self.pattern_encode = tf.reshape(pattern_encode_middle, shape=[1, cnn_hidden_size*3])

        # classifier ANN
        self.y_pre = classifier.pattern_classifier(self.pattern_encode, [cnn_hidden_size*3, 1])

        # loss
        self.pat_loss = tf.reduce_mean(tf.square(tf.cast(self.y, dtype=tf.float32) - tf.split(self.y_pre, 2, 1)[1]),
                                       name="pat_loss")
        self.tuple_loss = tf.reduce_mean(tf.square(tf.cast(self.t_y, dtype=tf.float32) - self.scores),
                                         name="tuple_loss")
        self.loss = self.pat_loss + balance * self.tuple_loss

        # prediction, accuracy
        self.prediction_result = tf.argmax(self.y_pre, axis=1, name="prediction_result")
        correct_prediction = tf.equal(self.y, tf.reshape(self.prediction_result, [1, 1]))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32), name="accuracy")

    def get_loss(self):
        return self.loss

    def get_accuracy(self):
        return self.accuracy



