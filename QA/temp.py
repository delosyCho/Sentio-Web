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
        self.whole_batch_index = 0
        self.Batch_Index = 0

        self.Batch_Size = 100
        self.Total_Batch_Size = 0

        self.P_Length = 70
        self.Q_Length = 85
        self.Word_Embedding_Dimension = 100

        self.argsort_length = []
        self.Paragraphs = []
        self.Questions = []
        self.Paragraphs_Length = []
        self.Questions_Length = []
        self.IDs = []
        self.Sentence_Index = []
        self.Sentence_s_e_Index = []

        self.in_path = "C:\\Users\\Administrator\\Desktop\\qadataset\\train-v1.1.json"
        self.data = json.load(open(self.in_path, 'r'))

        in_path_glove = "C:\\Users\\Administrator\\Desktop\\qadataset\\glove6B100d.txt"
        glove_f = codecs.open(in_path_glove, 'r', 'utf-8')

        self.words = []
        self.vectors = []

        for line in glove_f:
            tokens = line.split(' ')
            self.words.append(tokens.pop(0))
            self.vectors.append(tokens)

        self.vectors = numpy.array((self.vectors), 'f').reshape((-1, self.Word_Embedding_Dimension))

        self.dictionary = numpy.array(self.words)
        self.glove_arg_index = self.dictionary.argsort()
        self.dictionary.sort()

    def get_glove(self, word):
        word = "".join(word).lower()
        index = self.dictionary.searchsorted(word)
        # print("index,", index)

        if index == 400000:
            index = 0

        if word == self.dictionary[index]:
            # print("Success: ", word)
            # input()
            return self.vectors[self.dictionary.searchsorted(word)]
        else:
            # if str != '':
            #    print("fail: ", word)
            none_result = numpy.zeros((self.Word_Embedding_Dimension), dtype='f')
            for i in range(self.Word_Embedding_Dimension):
                none_result[i] = 10.0 / self.Word_Embedding_Dimension
            return none_result

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

    def get_json(self):
        return self.data

    def set_batch(self):
        myindex = 0
        numberOfQuestions = 0
        fileIndex = 0

        max_plength = -99
        max_qlength = -99

        sentence_index = 0

        for article in self.data['data']:
            for para in article['paragraphs']:
                for qa in para['qas']:
                    for answer in qa['answers']:
                        start_index = int(answer['answer_start'])
                        answer_length = len(answer['text'])

                        original_str = "".join(para['context'])

                        para_str = list(para['context'])
                        para_str[start_index] = '#'
                        para_str[start_index + answer_length - 1] = '#'

                        temp_str = "".join(para_str)
                        temp_str = temp_str.replace('.', ' .')
                        temp_str = temp_str.replace(',', ' ,')
                        temp_str = temp_str.replace('?', ' ?')
                        temp_str = temp_str.replace('!', ' !')
                        temp_str = temp_str.replace('(', ' ')
                        temp_str = temp_str.replace(')', ' ')
                        temp_str = temp_str.replace(u'\u2013', ' - ')
                        temp_str = temp_str.replace(u'\u2014', ' - ')
                        temp_str = temp_str.replace('-', ' - ')
                        temp_str = temp_str.replace('\'', ' \' ')
                        temp_str = temp_str.replace('\"', '')
                        # parapraph pre-processing

                        original_str = original_str.replace('.', ' .')
                        original_str = original_str.replace(',', ' ,')
                        original_str = original_str.replace('?', ' ?')
                        original_str = original_str.replace('!', ' !')
                        original_str = original_str.replace('(', ' ')
                        original_str = original_str.replace(')', ' ')
                        original_str = original_str.replace(u'\u2013', ' - ')
                        original_str = original_str.replace(u'\u2014', ' - ')
                        original_str = original_str.replace('-', ' - ')
                        original_str = original_str.replace('\'', ' \' ')
                        original_str = original_str.replace('\"', '')
                        # string for index setting

                        question_ = "".join(qa['question']).strip()
                        question_ = question_.replace('?', ' ?')
                        question_ = question_.split(' ')
                        # question pre-processing

                        Start_Index = 0
                        Stop_Index = 0

                        split1 = original_str.split(' ')

                        para_str = "".join(temp_str)
                        para_str = para_str.split(' ')

                        dot_count = 0
                        sentence_count = -1

                        for i in range(len(para_str)):
                            if para_str[i] == '.':
                                dot_count += 1

                            temp_list = list(para_str[i])

                            if len(temp_list) > 0:
                                if temp_list[0] == '#':
                                    sentence_count = dot_count
                        # start, stop index processing

                        self.IDs.append(qa['id'])
                        sentences = original_str.split('.').copy()

                        s_e_array = numpy.zeros(shape=[2], dtype=numpy.int32)
                        s_e_array[0] = sentence_index
                        index_array = numpy.zeros(shape=[len(original_str.split('.').copy())])
                        index_array[sentence_count] = 1

                        for i, sentence in enumerate(sentences):
                            self.Sentence_Index.append(index_array[i])
                            self.Paragraphs.append(sentence.split(' '))
                            self.Questions.append(question_.copy())

                            self.Paragraphs_Length.append(len(sentence.split(' ')))
                            self.Questions_Length.append(len(question_))
                            sentence_index += 1

                        s_e_array[1] = sentence_index
                        self.Sentence_s_e_Index.append((s_e_array.copy()))

                        if max_plength < len(split1):
                            max_plength = len(split1)
                        if max_qlength < len(question_):
                            max_qlength = len(question_)

        print("max p, q :", max_plength, max_qlength, " , ", numberOfQuestions)
        # self.P_Length = max_plength + 200
        # self.Q_Length = max_qlength + 50

        self.Total_Batch_Size = len(self.IDs)
        self.argsort_length = numpy.argsort(numpy.array(self.Paragraphs_Length))

        return max_plength, max_qlength

    def get_next_batch(self):
        cur_batch = 200
        cur_length = 70
        q_max = 50

        total_batch = 0

        np_range = numpy.arange(0, len(self.Sentence_s_e_Index), dtype='i')
        numpy.random.shuffle(np_range)

        for i in range(cur_batch):
            idx = np_range[i]
            total_batch += (self.Sentence_s_e_Index[idx][1] - self.Sentence_s_e_Index[idx][0])

        batch_paragraph = numpy.zeros((cur_batch, cur_length, self.Word_Embedding_Dimension), dtype='f')
        batch_question = numpy.zeros((cur_batch, q_max, self.Word_Embedding_Dimension), dtype='f')
        batch_label = numpy.zeros((cur_batch, 2), dtype='f')

        for i in range(cur_batch):
            idx = np_range[i]

            for j in range(self.Sentence_s_e_Index[idx][0], self.Sentence_s_e_Index[idx][1]):
                batch_label[i, int(self.Sentence_Index[j])] = 1
                batch_paragraph[i] = self.get_glove_sequence(length=cur_length, tokens=self.Paragraphs[j])
                batch_question[i] = self.get_glove_sequence(length=q_max, tokens=self.Questions[j])

        for i in range(cur_batch):
            idx = np_range[i]

        return batch_paragraph, batch_question, batch_label, 0, 0

    def get_test_batch(self):
        batch_paragraph_str = []
        batch_question_str = []
        index = 0

        line = self.Paragraphs[self.Batch_Index]
        q_line = self.Questions[self.Batch_Index]

        for j, sentence in enumerate(self.Paragraphs[self.Batch_Index]):
            TKs = "".join(sentence).split(' ')
            batch_paragraph_str.append(TKs)
            batch_question_str.append(self.Questions[self.Batch_Index])

            index += 1

        batch_paragraph = numpy.zeros((index, 70, self.Word_Embedding_Dimension), dtype='f')
        batch_question = numpy.zeros((index, 85, self.Word_Embedding_Dimension), dtype='f')

        for j, sentence in enumerate(self.Paragraphs[self.Batch_Index]):
            batch_paragraph[j] = self.get_glove_sequence(length=70, tokens=batch_paragraph_str[j])
            batch_question[j] = self.get_glove_sequence(length=85, tokens=batch_question_str[j])

        self.Batch_Index += 1

        return batch_paragraph, batch_question, self.Sentence_Index[self.Batch_Index - 1], line, q_line