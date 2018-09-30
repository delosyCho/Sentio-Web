import numpy
import codecs

import WIKI_QA_DataProcessor
import Sentence_Data_Processor
import BiCorpus_Dataset


class dataprocessor:
    def __init__(self):
        self.Word_Embedding_Dimension = 100
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

    def get_sequence_batch(self, data_index):
        if data_index == 0:



        batch_paragraph = numpy.zeros((total_batch, cur_length, self.Word_Embedding_Dimension), dtype='f')
        batch_question = numpy.zeros((total_batch, q_max, self.Word_Embedding_Dimension), dtype='f')
        batch_label = numpy.zeros((total_batch, 2), dtype='f')

        index = 0

        for i in range(self.Batch_Index, self.Batch_Index + cur_batch):
            idx = i

            for j in range(self.Start_Index[idx], self.Stop_Index[idx]):
                batch_label[index, int(self.Labels[j])] = 1
                batch_paragraph[index] = self.get_glove_sequence(length=cur_length, tokens=self.Paragraphs[j])
                batch_question[index] = self.get_glove_sequence(length=q_max, tokens=self.Questions[j])
                index += 1

        self.Batch_Index += cur_batch

        return batch_paragraph, batch_question, batch_label, np_range, cur_batch, self.Batch_Index