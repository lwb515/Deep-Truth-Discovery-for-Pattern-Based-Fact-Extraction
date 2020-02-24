# corpus
corpus = "news"

# models parameters:
# based tuple
based_t_crnn_max_length = 50
based_t_crnn_vec_dim = 120
based_t_extract_words = True

based_t_crnn_cnn_hidden_size = 128
based_t_crnn_cnn_kernel_size = 4
based_t_crnn_cnn_keep_prob = 0.76

based_t_crnn_rnn_num_units = 150
based_t_crnn_rnn_input_keep_prob = 1.0
based_t_crnn_rnn_output_keep_prob = 0.76


# based pattern
based_p_max_length = 5
based_p_vec_dim = 80
based_p_extract_words = True
with_truth_weight = True

based_p_cnn_hidden_size = 100
based_p_cnn_kernel_size = 3
based_p_cnn_stride_size = 1
based_p_cnn_keep_prob = 0.82

based_p_rnn_num_units = 85
based_p_rnn_input_keep_prob = 1.0
based_p_rnn_output_keep_prob = 0.82

balance = 1


# based type, model type. clstm -> cnn + lstm, cgru -> cnn + gru
# based_type = "tuple"
# based_model = "clstm"
# based_model = "cgru"

based_type = "pattern"
based_model = "lstm"
# based_model = "gru"


# common parameters
training_epoch = 100
learning_rate = 0.008
training_data_path = "data/" + corpus + "/training_data.txt"
test_data_path = "data/" + corpus + "/test_data.txt"


class ParameterInit:
    based_tuple_crnn = [based_t_crnn_max_length, based_t_crnn_vec_dim, based_t_crnn_cnn_hidden_size,
                        based_t_crnn_cnn_kernel_size, based_t_crnn_cnn_keep_prob, based_t_crnn_rnn_num_units,
                        based_t_crnn_rnn_input_keep_prob, based_t_crnn_rnn_output_keep_prob, based_t_extract_words]
    based_pattern = [based_p_max_length, based_p_vec_dim, based_p_cnn_hidden_size, based_p_cnn_kernel_size,
                     based_p_cnn_stride_size, based_p_cnn_keep_prob, based_p_rnn_num_units, based_p_rnn_input_keep_prob,
                     based_p_rnn_output_keep_prob, balance, based_p_extract_words, with_truth_weight]
    common_para = [training_epoch, learning_rate, training_data_path, test_data_path]
    model = based_model
    based_model = based_type

    def get_common_parameters(self):
        return self.common_para

    def get_parameter(self):
        if based_type == "tuple":
            return self.based_tuple_crnn
        else:
            return self.based_pattern

    def get_model_type(self):
        return self.model

    def get_based_type(self):
        return self.based_model
