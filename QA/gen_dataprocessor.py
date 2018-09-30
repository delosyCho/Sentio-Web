import tensorflow as tf
import json
import codecs
import numpy
import copy
import nltk


class Data_holder:
    def find_all(self, a_str, sub):
        start = 0
        while True:
            start = a_str.find(sub, start)
            if start == -1: return
            yield start
            start += len(sub)  # use start += 1 to find overlapping matches

    def __init__(self):
        """
        자동생성 위키 데이터를 불러오고 처리하기 위한 코드
        """

        """
        data load
        """
        path = ''


        """
        korean embedding
        """
        word2vec_kor = codecs.open('C:\\Users\\Administrator\\Desktop\\qadataset\\kor_word2vec_100d', 'r', 'utf-8')
        self.kor_words = []
        self.kor_vectors = []

        arr = []
        for i in range(100):
            pm = 1

            if i % 2 == 0:
                pm = -1

            arr.append(0.002 * pm * i)
        self.kor_words.append('#END')
        self.kor_vectors.append(arr)

        arr = []
        for i in range(100):
            pm = 1

            if i % 2 == 0:
                pm = 0.1
            elif i % 3 == 0:
                pm = -1

            arr.append(0.1 * pm)
        self.kor_words.append('#START')
        self.kor_vectors.append(arr)

        for line in word2vec_kor:
            tokens = line.split('\t')
            self.kor_words.append(tokens.pop(0))
            self.kor_vectors.append(tokens)

        print(self.kor_words[0])
        print(self.kor_words[1])

        self.kor_dictionary = numpy.array(self.kor_words)
        self.word2vec_arg_index = self.kor_dictionary.argsort()
        self.kor_dictionary.sort()

        print("word2vec ready")

        """
        bicorpus
        """
        eng_corpus = open('korean-english-park.train.en', 'r', encoding='utf-8')
        eng_text = eng_corpus.read()
        eng_text = eng_text.replace('\"', '').replace('(', ' ').replace(')', ' ')
        eng_text = eng_text.replace('  ', ' ').replace('  ', ' ').replace('  ', ' ')
        lines_eng = eng_text.split('\n')


        kor_corpus = open('korean-english-park.train.ko', 'r', encoding='utf-8')
        kor_text = kor_corpus.read()
        kor_text = kor_text.replace('\"', '').replace('(', ' ').replace(')', ' ')
        kor_text = kor_text.replace('  ', ' ').replace('  ', ' ').replace('  ', ' ')
        lines_ko = kor_text.split('\n')

        self.eng_sentences = []
        self.kor_sentences = []

        for i in range(len(lines_ko)):
            if lines_ko[i] != '어휘 :':
                self.eng_sentences.append(lines_eng[i].split())
                self.kor_sentences.append(lines_ko[i].split())

        eng_corpus.close()
        kor_corpus.close()

    def get_dic_index(self, word):
        word = "".join(word).lower()
        index = self.dictionary.searchsorted(word)

        if index == 400000:
            index = 0

        if word == self.dictionary[index]:
            # print("Success: ", word)
            # input()
            return index
        else:
            return 0

    def get_dic_index_kor(self, word):
        word = "".join(word).lower()
        index = self.kor_dictionary.searchsorted(word)

        if index == 400000:
            index = 0

        if word == self.kor_dictionary[index]:
            # print("Success: ", word)
            # input()
            return index
        else:
            return 0

    def get_glove(self, word):
        word = "".join(word).lower()
        index = self.dictionary.searchsorted(word)
        # print("index,", index)

        if index == 400000:
            index = 0

        if word == self.dictionary[index]:
            # print("Success: ", word)
            # input()
            return self.vectors[self.glove_arg_index[index]]
        else:
            # if str != '':
            #    print("fail: ", word)
            none_result = numpy.zeros((self.Word_Embedding_Dimension), dtype='f')
            for i in range(self.Word_Embedding_Dimension):
                none_result[i] = 10.0 / self.Word_Embedding_Dimension
                return self.vectors[0]

    def get_glove_sequence(self, length, tokens):
        result = numpy.zeros((length, self.Word_Embedding_Dimension), dtype='f')
        padding = 0
        mylength = length
        if len(tokens) < length:
            mylength = len(tokens)
            padding = length - mylength
        for i in range(mylength):
            result[padding + i] = self.get_glove(tokens[i])

        return result

    def get_glove_kor(self, word):
        #word = "".join(word).lower()
        index = self.kor_dictionary.searchsorted(word)
        # print("index,", index)

        if index == self.kor_dictionary.shape[0] or index == -1:
            index = 0

        if word == self.kor_dictionary[index]:
            # print("Success: ", word)
            # input()
            return self.kor_vectors[self.word2vec_arg_index[index]]
        else:
            # if str != '':
            #    print("fail: ", word)

            return self.kor_vectors[0]

    def get_glove_sequence_kor(self, length, tokens):
        result = numpy.zeros((length, self.Word_Embedding_Dimension), dtype='f')
        padding = 0
        mylength = length
        if len(tokens) < length:
            mylength = len(tokens)
            padding = length - mylength
        for i in range(mylength):
            result[padding + i] = self.get_glove_kor(tokens[i])

        return result

    def get_next_batch(self):
        cur_batch = 32
        cur_length = 50

        np_range = numpy.arange(0, len(self.eng_sentences), dtype='i')
        numpy.random.shuffle(np_range)

        batch_kor_sentences = numpy.zeros((cur_batch, cur_length), dtype='f')
        batch_eng_sentences = numpy.zeros((cur_batch, cur_length), dtype='f')
        batch_dump = numpy.zeros((cur_batch), dtype=numpy.int32)

        token_index = self.get_dic_index('#END')
        for i in range(cur_batch):
            for j in range(50):
                batch_eng_sentences[j] = token_index

        token_index = self.get_dic_index_kor('#END')
        for i in range(cur_batch):
            for j in range(50):
                batch_kor_sentences[j] = token_index

        for i in range(cur_batch):
            idx = np_range[i]

            for j in range(len(self.kor_sentences[idx])):
                batch_kor_sentences[i, j] = self.get_dic_index_kor(self.kor_sentences[idx][j])
            for j in range(len(self.kor_sentences[idx])):
                batch_eng_sentences[i, j] = self.get_dic_index(self.eng_sentences[idx][j])

        return batch_kor_sentences, batch_eng_sentences, batch_dump

    def QA_Test_Batch(self):
        idx = self.Test_Batch_index
        try:
            labels_start = int(self.Labels_index[idx].split('#')[0]) + 1
        except:
            labels_start = 1
        labels_stop = int(self.Labels_index[idx].split('#')[1])

        cur_batch = labels_stop - labels_start
        cur_length = 50

        cur_batch_sen = []
        cur_batch_que = []
        cur_batch_lab = []

        cur_batch_sen_sample = []
        cur_batch_que_sample = []

        for i in range(labels_start, labels_stop):
            try:
                cur_batch_sen.append(self.Sentences[i].strip().split(' '))
                cur_batch_que.append(self.Questions[i].strip().split(' '))
                cur_batch_lab.append(int(self.Labels[i]))

                cur_batch_que_sample.append(self.Questions[i])
                cur_batch_sen_sample.append((self.Sentences[i]))
            except:
                print(labels_start, labels_stop, idx)
                print(len(self.Sentences))
                print(len(self.Questions))

        batch_sentences = numpy.zeros((cur_batch, cur_length, self.Word_Embedding_Dimension), dtype='f')
        batch_questions = numpy.zeros((cur_batch, cur_length, self.Word_Embedding_Dimension), dtype='f')
        batch_label = numpy.zeros((cur_batch, 2), dtype='i')
        batch_dump = numpy.zeros((cur_batch), dtype=numpy.int32)

        for i in range(cur_batch):
            batch_sentences[i] = self.get_glove_sequence_kor(length=cur_length, tokens=cur_batch_sen[i])
            batch_questions[i] = self.get_glove_sequence_kor(length=cur_length, tokens=cur_batch_que[i])
            batch_label[i, int(cur_batch_lab[i])] = 1

        self.Test_Batch_index += 1

        return batch_sentences, batch_questions, batch_label, cur_batch_sen_sample, cur_batch_que_sample, batch_dump
