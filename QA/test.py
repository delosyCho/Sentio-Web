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

        self.P_Length = 850
        self.Q_Length = 85
        self.Word_Embedding_Dimension = 100

        self.argsort_length = []
        self.Paragraphs = []
        self.Questions = []
        self.Paragraphs_Length = []
        self.Questions_Length = []
        self.IDs = []
        self.start_index_batch = []
        self.stop_index_batch = []

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
                none_result[i] = 3.0 / self.Word_Embedding_Dimension
            return none_result

    def get_glove_sequence(self, length, tokens):
        result = numpy.zeros((length, self.Word_Embedding_Dimension), dtype='f')
        mylength = length
        if len(tokens) < length:
            mylength = len(tokens)
        for i in range(mylength):
            result[i] = self.get_glove(tokens[i])

        return result

    def get_json(self):
        return self.data

    def set_batch(self):
        myindex = 0
        numberOfQuestions = 0
        fileIndex = 0

        max_plength = -99
        max_qlength = -99

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

                        for i in range(len(para_str)):
                            temp_list = list(para_str[i])

                            if len(temp_list) > 0:
                                if temp_list[0] == '#':
                                    Start_Index = i
                                if temp_list[len(temp_list) - 1] == '#':
                                    Stop_Index = i
                        # start, stop index processing

                        self.start_index_batch.append(Start_Index)
                        self.stop_index_batch.append(Stop_Index)
                        self.IDs.append(qa['id'])

                        self.Paragraphs.append(split1.copy())
                        self.Questions.append(question_.copy())
                        self.Paragraphs_Length.append(len(split1))
                        self.Questions_Length.append(len(question_))

                        if max_plength < len(split1):
                            max_plength = len(split1)
                        if max_qlength < len(question_):
                            max_qlength = len(question_)

        print("max p, q :", max_plength, max_qlength, " , ", numberOfQuestions)
        # self.P_Length = max_plength + 200
        # self.Q_Length = max_qlength + 50
        self.Total_Batch_Size = len(self.start_index_batch)
        self.Total_Batch_Size = len(self.IDs)

        self.argsort_length = numpy.argsort(numpy.array(self.Paragraphs_Length))

        return max_plength, max_qlength

    def get_next_batch(self):
        size_shape = [0, self.P_Length, 15]

        cur_batch = 10

        while True:
            if cur_batch + self.Batch_Index > self.Total_Batch_Size:
                cur_batch = self.Total_Batch_Size - self.Batch_Index
                break

            size = cur_batch * self.Paragraphs_Length[self.argsort_length[cur_batch + self.Batch_Index]]
            if size < 100 * 1000:
                cur_batch += 5
            else:
                break

        size_shape[0] = self.Total_Batch_Size - self.Batch_Index * self.Batch_Size

        p_max = 0
        q_max = 0
        for i in range(self.Batch_Index, self.Batch_Index + cur_batch):
            if p_max < self.Paragraphs_Length[self.argsort_length[i]]:
                p_max = self.Paragraphs_Length[self.argsort_length[i]]
            if q_max < self.Questions_Length[self.argsort_length[i]]:
                q_max = self.Questions_Length[self.argsort_length[i]]

        batch_ = self.Total_Batch_Size - (self.Batch_Index * self.Batch_Size)

        batch_paragraph = numpy.zeros((cur_batch, p_max, self.Word_Embedding_Dimension), dtype='f')
        batch_question = numpy.zeros((cur_batch, q_max, self.Word_Embedding_Dimension), dtype='f')
        batch_start_index = numpy.zeros((cur_batch, p_max, 1), dtype='f')
        batch_stop_index = numpy.zeros((cur_batch, p_max, 1), dtype='f')
        batch_start_index_value = numpy.zeros((cur_batch), dtype=numpy.int32)
        batch_stop_index_value = numpy.zeros((cur_batch), dtype=numpy.int32)
        batch_IDs = numpy.zeros((cur_batch), dtype='<U30')
        batch_paragraph_str = []
        batch_question_str = []

        index = 0
        for i in range(self.Batch_Index, self.Batch_Index + cur_batch):
            batch_paragraph_str.append(self.Paragraphs[self.argsort_length[i]])
            batch_question_str.append(self.Questions[self.argsort_length[i]])

            batch_paragraph[index] = self.get_glove_sequence(length=p_max, tokens=self.Paragraphs[self.argsort_length[i]])
            batch_question[index] = self.get_glove_sequence(length=q_max, tokens=self.Questions[self.argsort_length[i]])
            batch_start_index[index, int(self.start_index_batch[self.argsort_length[i]])] = 1
            batch_stop_index[index, int(self.stop_index_batch[self.argsort_length[i]])] = 1
            batch_start_index_value[index] = int(self.start_index_batch[self.argsort_length[i]])
            batch_stop_index_value[index] = int(self.stop_index_batch[self.argsort_length[i]])
            batch_IDs[index] = str(self.IDs[self.argsort_length[i]])
            #테스트 하기 위한 paragraph, question raw 리턴 해보셈 임시로
            index += 1

        self.Batch_Index += cur_batch
        if self.Batch_Index >= self.Total_Batch_Size:
            self.Batch_Index = 0

        return batch_paragraph, batch_question, batch_start_index, batch_stop_index, batch_start_index_value, batch_stop_index_value, batch_paragraph_str, batch_question_str
