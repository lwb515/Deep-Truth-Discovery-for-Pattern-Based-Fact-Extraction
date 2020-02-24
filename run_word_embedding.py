from gensim.models import Word2Vec
from gensim.test.utils import common_texts
from para_init import corpus as corpus_type
from para_init import word2vec_dim


class WordEmbeddor:
    def __remove_label(self, text):
        words = text.split(" ")
        result, i = [], 0
        while i < len(words):
            if words[i].find("<") == -1 and words[i].find(">") == -1:
                result.append(words[i])
                i += 1
            else:
                label_start = words[i].find("<")
                text_start = words[i].find(">")
                label_end = words[i].find("</")
                if label_start != -1 and label_end != -1:
                    result.append(words[i][text_start + 1:label_end])
                    i += 1
                    continue
                if label_start != -1 and label_end == -1:
                    word = words[i][text_start + 1:] + " "
                    while True:
                        i = i + 1
                        index = words[i].find("</")
                        if index == -1:
                            word += words[i] + " "
                        else:
                            word += words[i][:index]
                            break
                    result.append(word)
                    i += 1
                    continue
                i += 1
        return result

    def word_embedding(self, size=based_p_vec_dim, corpus_file="data/" + corpus_type + "/corpus.txt",
                       model_file="models/word_embedding/" + corpus_type + "/word2vec.model",
                       batch_size=10000):
        model = Word2Vec(common_texts, min_count=1, size=size)
        with open(corpus_file, 'r', encoding="UTF-8") as f:
            lines = []
            line_count, batch_count = 0, 0
            for line in f:
                lines.append(self.__remove_label(line))
                line_count += 1
                if line_count == batch_size:
                    model.build_vocab(lines, update=True)
                    model.train(lines, total_examples=model.corpus_count, epochs=model.iter)
                    lines.clear()
                    line_count = 0
                    batch_count += 1
        model.save(model_file)


if __name__ == '__main__':
    model = WordEmbeddor()
    model.word_embedding(size=word2vec_dim)
