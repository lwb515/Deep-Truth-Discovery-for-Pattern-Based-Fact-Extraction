import os
import time
import copy
import tensorflow as tf
import para_init as pi
import base_pattern.joint_model as joint_model
import base_tuple.cnn_rnn_model as based_tuple_crnn
from para_init import corpus as corpus_type


class Demo:
    def __init__(self):
        para_init = copy.deepcopy(pi.ParameterInit())
        self.training_epoch, self.learning_rate, self.train_data_file, self.test_data_file\
            = copy.deepcopy(para_init.get_common_parameters())
        self.model_para = copy.deepcopy(para_init.get_parameter())
        self.based_type = copy.deepcopy(para_init.get_based_type())
        self.model_type = copy.deepcopy(para_init.get_model_type())
        self.saver_file_path = "models/" + corpus_type + "/base_" + self.based_type + "/classifier/"
        self.test_output_file = "data/result/" + corpus_type + "/base_" + self.based_type \
                                + "_result/" + self.model_type

    def train(self):
        if self.based_type == "pattern":
            self.model_para.append(self.model_type)
            new_model = joint_model.Model(self.train_data_file, self.model_para)
        else:
            if self.model_type == "clstm":
                self.model_para.append("lstm")
            else:
                self.model_para.append("gru")
            new_model = based_tuple_crnn.Model(self.train_data_file, self.model_para)

        # optimizer
        optimizer = tf.train.AdadeltaOptimizer(self.learning_rate).minimize(loss=new_model.loss, name="min_optimizer")

        run_arrays = [optimizer, new_model.loss, new_model.accuracy]
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            saver = tf.train.Saver()

            print("start train...")
            for epoch in range(0, self.training_epoch):
                __loss, __acc = [], []
                print("EPOCH=" + str(epoch))
                for j in range(len(new_model.patterns)):
                    fedict = {
                        new_model.y: new_model.y_info[j],
                        new_model.e: new_model.x_e[j],
                        new_model.a: new_model.x_a[j],
                        new_model.v: new_model.x_v[j],
                        new_model.f: new_model.pat_freq[j],
                        new_model.f_tot: new_model.freq_tot[j]
                    }
                    if self.based_type == "pattern":
                        fedict[new_model.t_y] = new_model.tuple_labels[j]

                    _, _loss, _accuracy = sess.run(
                        fetches=run_arrays,
                        feed_dict=fedict
                    )
                    __loss.append(_loss)
                    __acc.append(_accuracy)
                __acc = sum(__acc) / len(__acc)

            print("save model....")
            saver.save(sess, self.saver_file_path + self.model_type + "/" + self.model_type)

    def test(self):
        if self.based_type == "pattern":
            new_model = joint_model.Model(self.test_data_file, self.model_para)
        else:
            new_model = based_tuple_crnn.Model(self.test_data_file, self.model_para)

        print("start test...")
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(
                os.path.join(self.saver_file_path + self.model_type, self.model_type + '.meta'))
            saver.restore(sess, tf.train.latest_checkpoint(self.saver_file_path + self.model_type + "/"))

            if self.based_type == "pattern":
                run_arrays = [new_model.accuracy, new_model.prediction_result, new_model.tuple_acc, new_model.scores]
                start = time.clock()
                __acc, __pred, __tacc, __tuples_score = [], [], [], []

                for j in range(len(new_model.patterns)):
                    _accuracy, _pred, _tacc, _t_score = sess.run(
                        fetches=run_arrays,
                        feed_dict={
                            new_model.y: new_model.y_info[j],
                            new_model.e: new_model.x_e[j],
                            new_model.a: new_model.x_a[j],
                            new_model.v: new_model.x_v[j],
                            new_model.f: new_model.pat_freq[j],
                            new_model.f_tot: new_model.freq_tot[j],
                            new_model.t_y: new_model.tuple_labels[j]
                        }
                    )
                    __acc.append(_accuracy)
                    __pred.append(_pred[0])
                    __tacc.append(_tacc)
                    __tuples_score.append(_t_score)

                self.based_pat_output_tuples_result(new_model.tuple_info, __tuples_score, new_model.patterns)
                self.based_pat_output_result(__pred, new_model.y_info, new_model.patterns)

                end = time.clock()
                print("cost time:" + str(end - start))
            else:
                run_arrays = [new_model.accuracy, new_model.prediction_result, new_model.tuple_score]

                start = time.clock()
                __acc, __pred, __ts = [], [], []
                for j in range(len(new_model.patterns)):
                    _accuracy, _pred, _ts = sess.run(
                        fetches=run_arrays,
                        feed_dict={
                            new_model.y: new_model.y_info[j],
                            new_model.e: new_model.x_e[j],
                            new_model.a: new_model.x_a[j],
                            new_model.v: new_model.x_v[j],
                            new_model.f: new_model.pat_freq[j],
                            new_model.f_tot: new_model.freq_tot[j]
                        }
                    )
                    __acc.append(_accuracy)
                    __pred.append(_pred)
                    __ts.append(_ts)

                self.based_tuple_output_result(__pred, new_model.patterns, new_model.pat_infos)
                self.cal_pattern_label(__ts, new_model.pat_labels, new_model.pat_freq,
                                                 new_model.freq_tot, new_model.patterns)

                end = time.clock()
                print("cost time" + str(end - start))

    def cal_pattern_label(self, tuples_score, pat_label, tuples_freq, tot_freq, patterns):
        result = []
        for i in range(len(tuples_score)):
            score = 0
            for j in range(len(tuples_score[i])):
                score += tuples_score[i][j] * tuples_freq[i][j] / tot_freq[i]
            result.append(score)

        file = self.test_output_file + "_pat_result.txt"
        with open(file, "w") as f:
            for i in range(len(patterns)):
                f.write(patterns[i] + "\t" + str(pat_label[i]) + "\t" + str(result[i][0]) + "\n")
        print("gen tuples result")

        acc = 0
        for i in range(len(result)):
            if result[i] >= 0.5 and pat_label[i] == 1:
                acc += 1
            if result[i] < 0.5 and pat_label[i] == 0:
                acc += 1
        acc = acc / len(result)
        return acc

    def based_pat_output_result(self, pred, real, patterns):
        with open(self.test_output_file + "_pat_result.txt", "w") as f:
            for i in range(len(patterns)):
                f.write(patterns[i] + "\t" + str(pred[i]) + "\t" + str(real[i]) + "\n")

        print("gen pats result...")

    def based_pat_output_tuples_result(self, tuple, score, patterns):
        file = self.test_output_file + "_tuple_result.txt"
        with open(file, "w") as f:
            for i in range(len(tuple)):
                for j in range(len(tuple[i])):
                    f.write(patterns[i].replace("\n", "") + "\t")
                    for k in range(len(tuple[i][j])):
                        stri = tuple[i][j][k].replace("\n", "")
                        f.write(stri + "\t")

                    f.write(str(score[i][j][0]) + "\n")

        print("gen tuples result...")

    def based_tuple_output_result(self, pred, patterns, tuples):
        with open(self.test_output_file + "_tuple_result.txt", "w") as f:
            for i in range(len(patterns)):
                for j in range(len(tuples[i])):
                    f.write(patterns[i] + "\t")
                    for attr in tuples[i][j]:
                        attr = attr.replace("\n", "")
                        f.write(attr + "\t")
                    f.write(str(pred[i][j]) + "\n")

        print("gen result...")


if __name__ == '__main__':
    demo = Demo()
    demo.train()
    demo.test()





