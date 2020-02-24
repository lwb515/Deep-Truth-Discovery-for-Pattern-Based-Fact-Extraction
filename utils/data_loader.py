from gensim.models import KeyedVectors
import numpy as np
from para_init import corpus as corpus_type


def get_words_vec(model, words, max_length=50, attr=False, rnn=False, flag=False, vec_dim=100):
    if rnn:
        vec = np.zeros((1, vec_dim), dtype=np.float32)
        try:
            vec[0] = model[words]
        except KeyError:
            vec[0] = np.random.normal(size=[1, vec_dim])
        return vec

    vec = np.zeros((max_length, vec_dim), dtype=np.float32)
    if attr is False:
        try:
            vec[0] = model[words]
        except KeyError:
            vec[0] = np.random.normal(size=[1, vec_dim])
        return vec
    else:
        words = words.split(" ")
        for i in range(len(words)):
            try:
                vec[i] = model[words[i]]
            except KeyError:
                vec[i] = np.random.normal(size=[1, vec_dim])
        if flag:
            words_flag = True
            if len(words) == 1:
                words_flag = False
            return vec, words_flag
        return vec


def get_patterns_label(file_path="data/" + corpus_type + "/patterns_label.txt"):
    patterns_label = {}
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            attr1 = line.split("\t")[0]
            try:
                attr2 = int(line.split("\t")[1].replace("\n", ""))
                patterns_label[attr1] = attr2
            except ValueError:
                patterns_label[attr1] = 0
    return patterns_label


def read_pattern_tuples(file_path):
    patterns = []
    patterns_info = []
    with open(file_path, 'r') as f:
        f.readline()
        lines = f.readlines()
        for line in lines:
            info = line.split("\t")
            if info[0] in patterns:
                index = patterns.index(info[0])
                patterns_info[index].append(info[1:])
            else:
                patterns.append(info[0])
                patterns_info.append([info[1:]])

    return patterns, patterns_info


def get_pattern_info_vec(file_path, vec_dim=100, rnn=False, max_length=50, flag=False,
                         wordvec_path="models/word_embedding/" + corpus_type + "/word2vec.model",):
    patterns, pat_infos = read_pattern_tuples(file_path)
    pat_labels = get_patterns_label()
    value_vectors, entity_vectors, attr_vectors, freq_info, freq_tot, label, pats_label = [], [], [], [], [], [], []
    model_wv = KeyedVectors.load(wordvec_path, mmap='r')
    bad_patterns = []
    for i in range(len(patterns)):
        freq = 0
        entity_vectors.append([])
        value_vectors.append([])
        attr_vectors.append([])
        freq_info.append([])
        label.append([])
        value_error_tuples = []
        if flag:
            one_word_tuples = []
        for tinfo in pat_infos[i]:
            try:
                freq = freq + int(tinfo[3])
                if flag:
                    attr_vec, words_flag = get_words_vec(model_wv, tinfo[0], max_length, True, rnn, flag, vec_dim=vec_dim)
                    if words_flag is False:
                        one_word_tuples.append(tinfo)
                        continue
                else:
                    attr_vec = get_words_vec(model_wv, tinfo[0], max_length, True, rnn, flag, vec_dim=vec_dim)
                entity_vec = get_words_vec(model_wv, tinfo[1], max_length, False, rnn, vec_dim=vec_dim)
                val_vec = get_words_vec(model_wv, tinfo[2], max_length, False, rnn, vec_dim=vec_dim)

                entity_vectors[-1].append(entity_vec)
                value_vectors[-1].append(val_vec)
                attr_vectors[-1].append(attr_vec)
                freq_info[-1].append([int(tinfo[3])])
                label[-1].append([int(tinfo[6])])
            except ValueError:
                value_error_tuples.append(tinfo)

        for value_error_tuple in value_error_tuples:
            pat_infos[i].remove(value_error_tuple)

        if flag:
            for one_word_tuple in one_word_tuples:
                pat_infos[i].remove(one_word_tuple)

        if len(entity_vectors[-1]) == 0 or len(value_vectors[-1]) == 0 or len(attr_vectors[-1]) == 0:
            bad_patterns.append(patterns[i])
            entity_vectors.pop()
            value_vectors.pop()
            attr_vectors.pop()
            freq_info.pop()
            label.pop()
            continue

        entity_vectors[-1] = np.array(entity_vectors[-1])
        value_vectors[-1] = np.array(value_vectors[-1])
        attr_vectors[-1] = np.array(attr_vectors[-1])
        freq_info[-1] = np.array(freq_info[-1])
        label[-1] = np.array(label[-1])
        freq_tot.append(freq)

    for pattern in bad_patterns:
        index = patterns.index(pattern)
        pat_infos.pop(index)
        patterns.pop(index)

    for pattern in patterns:
        if pattern in pat_labels:
            pats_label.append(pat_labels[pattern])
        else:
            pats_label.append(0)

    patterns = np.array(patterns)
    tuple_vec = np.array([np.array(entity_vectors), np.array(attr_vectors), np.array(value_vectors)])
    freq_info = np.array(freq_info)
    label = np.array(label)
    freq_tot = np.array(freq_tot)

    return patterns, tuple_vec, freq_info, label, freq_tot, pat_infos, pats_label
    # f_path = "../data/middle_result/pattern_info/" + ftype
    # np.save(f_path + "_patterns", patterns)
    # np.save(f_path + "_tuple_vec", tuple_vec)
    # np.save(f_path + "_freq_info", freq_info)
    # np.save(f_path + "_label", label)
    # np.save(f_path + "_freq_tot", freq_tot)


def get_pattern_info_vec2(file_path, vec_dim=100, max_length=50, flag=False,
                          wordvec_path="models/word_embedding/" + corpus_type + "/word2vec.model"):
    patterns, pat_infos = read_pattern_tuples(file_path)
    pat_labels = get_patterns_label()
    value_vectors, entity_vectors, attr_vectors, freq_info, freq_tot, label, tuple_labels, tuple_info \
        = [], [], [], [], [], [], [], []
    model_wv = KeyedVectors.load(wordvec_path, mmap='r')
    bad_patterns = []
    for i in range(len(patterns)):
        freq = 0
        entity_vectors.append([])
        value_vectors.append([])
        attr_vectors.append([])
        freq_info.append([])
        tuple_labels.append([])
        tuple_info.append([])
        value_error_tuples = []
        if flag:
            one_word_tuples = []

        for tinfo in pat_infos[i]:
            try:
                freq = freq + int(tinfo[3])
                if flag:
                    attr_vec, words_flag = get_words_vec(model_wv, tinfo[0], max_length, True, False, flag, vec_dim=vec_dim)
                    if words_flag is False:
                        one_word_tuples.append(tinfo)
                        continue
                else:
                    attr_vec = get_words_vec(model_wv, tinfo[0], max_length, True, False, flag, vec_dim=vec_dim)

                entity_vec = get_words_vec(model_wv, tinfo[1], max_length, vec_dim=vec_dim)
                val_vec = get_words_vec(model_wv, tinfo[2], max_length, vec_dim=vec_dim)

                entity_vectors[-1].append(entity_vec)
                value_vectors[-1].append(val_vec)
                attr_vectors[-1].append(attr_vec)
                tuple_info[-1].append(tinfo)

                freq_info[-1].append([int(tinfo[3])])
                tuple_labels[-1].append([int(tinfo[6])])
            except ValueError:
                value_error_tuples.append(tinfo)

        for value_error_tuple in value_error_tuples:
            pat_infos[i].remove(value_error_tuple)
        if flag:
            for one_word_tuple in one_word_tuples:
                pat_infos[i].remove(one_word_tuple)

        if len(entity_vectors[-1]) == 0 or len(value_vectors[-1]) == 0 or len(attr_vectors[-1]) == 0:
            bad_patterns.append(patterns[i])
            entity_vectors.pop()
            value_vectors.pop()
            attr_vectors.pop()
            freq_info.pop()
            tuple_labels.pop()
            tuple_info.pop()
            continue
        entity_vectors[-1] = np.array(entity_vectors[-1])
        value_vectors[-1] = np.array(value_vectors[-1])
        attr_vectors[-1] = np.array(attr_vectors[-1])
        freq_info[-1] = np.array(freq_info[-1])
        tuple_labels[-1] = np.array(tuple_labels[-1])
        freq_tot.append(freq)

    for pattern in bad_patterns:
        patterns.remove(pattern)

    label.clear()
    for pattern in patterns:
        if pattern in pat_labels:
            label.append(pat_labels[pattern])
        else:
            label.append(0)

    patterns = np.array(patterns)
    tuple_vec = np.array([np.array(entity_vectors), np.array(attr_vectors), np.array(value_vectors)])
    freq_info = np.array(freq_info)
    tuple_labels = np.array(tuple_labels)
    label = np.array(label)
    freq_tot = np.array(freq_tot)
    return patterns, tuple_vec, freq_info, label, freq_tot, tuple_labels, tuple_info


def load_info(ftype):
    f_path = "data/middle_result/pattern_info/" + ftype
    patterns = np.load(f_path + "_patterns.npy")
    tuple_vec = np.load(f_path + "_tuple_vec.npy")
    freq_info = np.load(f_path + "_freq_info.npy")
    label = np.load(f_path + "_label.npy")
    freq_tot = np.load(f_path + "_freq_tot.npy")
    return patterns, tuple_vec, freq_info, label, freq_tot


