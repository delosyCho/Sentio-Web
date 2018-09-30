import WIKI_QA_DataProcessor
import Refined_DataProcessor
import numpy
import codecs


class Model:
    def __init__(self):
        self.Word_Embedding_Dimension = 100

        self.WDP = Refined_DataProcessor.Refined_DataProcessor(isSQUAD=False)
        self.RDP = Refined_DataProcessor.Refined_DataProcessor()

        self.batch_index1 = len(self.WDP.Start_Index) - 200
        self.batch_index2 = len(self.RDP.Start_Index) - 5000

        """
        glove 100dimension pre-trained model loading
        """
        in_path_glove = "C:\\Users\\Administrator\\Desktop\\qadataset\\glove6B100d.txt"
        glove_f = codecs.open(in_path_glove, 'r', 'utf-8')

        self.words = []
        self.vectors = []

        arr = []
        for i in range(100):
            pm = 1

            if i % 2 == 0:
                pm = -1

            arr.append(0.002 * pm * i)
        self.words.append('#END')
        self.vectors.append(arr)

        arr = []
        for i in range(100):
            pm = 1

            if i % 2 == 0:
                pm = 0.1
            elif i % 3 == 0:
                pm = -1

            arr.append(0.1 * pm)
        self.words.append('#START')
        self.vectors.append(arr)

        for line in glove_f:
            tokens = line.split(' ')
            self.words.append(tokens.pop(0))
            self.vectors.append(tokens)

        self.vectors = numpy.array((self.vectors), 'f').reshape((-1, self.Word_Embedding_Dimension))

        self.dictionary = numpy.array(self.words)
        self.glove_arg_index = self.dictionary.argsort()
        self.dictionary.sort()
        ########################

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

    def get_test_batch(self):
        batch_size = 1
        batch_size2 = 1

        p_length = 70
        q_length = 50

        np_range = numpy.arange(0, len(self.WDP.Start_Index) - 200, dtype='i')
        numpy.random.shuffle(np_range)

        np_range2 = numpy.arange(0, len(self.RDP.Start_Index) - 5000, dtype='i')
        numpy.random.shuffle(np_range2)

        paragraphs = []
        questions = []
        labels = []

        for i in range(batch_size):
            idx = np_range[i]

            for j in range(self.WDP.Start_Index[idx], self.WDP.Stop_Index[idx]):
                paragraphs.append(self.WDP.Paragraphs[j])
                questions.append(self.WDP.Questions[j])
                labels.append(self.WDP.Labels[j])

        for i in range(batch_size2):
            idx = np_range2[i]

            for j in range(self.RDP.Start_Index[idx], self.RDP.Stop_Index[idx]):
                paragraphs.append(self.RDP.Paragraphs[j])
                questions.append(self.RDP.Questions[j])
                labels.append(self.RDP.Labels[j])

        total_batch = 0

        for i in range(batch_size):
            idx = np_range[i]
            total_batch += (self.WDP.Stop_Index[idx] - self.WDP.Start_Index[idx])
        for i in range(batch_size2):
            idx = np_range2[i]
            total_batch += (self.RDP.Stop_Index[idx] - self.RDP.Start_Index[idx])

        batch_paragraph = numpy.zeros((total_batch, p_length, self.Word_Embedding_Dimension), dtype='f')
        batch_question = numpy.zeros((total_batch, q_length, self.Word_Embedding_Dimension), dtype='f')
        batch_label = numpy.zeros((total_batch, 2), dtype='f')

        index = 0

        for i in range(batch_size):
            idx = np_range[i]

            for j in range(self.WDP.Start_Index[idx], self.WDP.Stop_Index[idx]):
                batch_label[index, int(self.WDP.Labels[j])] = 1
                batch_paragraph[index] = self.get_glove_sequence(length=p_length, tokens=self.WDP.Paragraphs[j])
                batch_question[index] = self.get_glove_sequence(length=q_length, tokens=self.WDP.Questions[j])
                index += 1

        for i in range(batch_size2):
            idx = np_range2[i]

            for j in range(self.RDP.Start_Index[idx], self.RDP.Stop_Index[idx]):
                batch_label[index, int(self.RDP.Labels[j])] = 1
                batch_paragraph[index] = self.get_glove_sequence(length=p_length, tokens=self.RDP.Paragraphs[j])
                batch_question[index] = self.get_glove_sequence(length=q_length, tokens=self.RDP.Questions[j])
                index += 1

        return batch_paragraph, batch_question, batch_label, 0, 0

    def get_next_batch_index(self):
        batch_size = 0
        batch_size2 = 32

        p_length = 50
        q_length = 50

        np_range = numpy.arange(0, len(self.WDP.Start_Index) - 200, dtype='i')
        numpy.random.shuffle(np_range)

        np_range2 = numpy.arange(0, len(self.RDP.Start_Index) - 5000, dtype='i')
        numpy.random.shuffle(np_range2)

        paragraphs = []
        questions = []
        labels = []

        for i in range(batch_size):
            idx = np_range[i]

            for j in range(self.WDP.Start_Index[idx], self.WDP.Stop_Index[idx]):
                paragraphs.append(self.WDP.Paragraphs[j])
                questions.append(self.WDP.Questions[j])
                labels.append(self.WDP.Labels[j])

        for i in range(batch_size2):
            idx = np_range2[i]

            for j in range(self.RDP.Start_Index[idx], self.RDP.Stop_Index[idx]):
                paragraphs.append(self.RDP.Paragraphs[j])
                questions.append(self.RDP.Questions[j])
                labels.append(self.RDP.Labels[j])

        total_batch = 0

        for i in range(batch_size):
            idx = np_range[i]
            total_batch += (self.WDP.Stop_Index[idx] - self.WDP.Start_Index[idx])
        for i in range(batch_size2):
            idx = np_range2[i]
            total_batch += (self.RDP.Stop_Index[idx] - self.RDP.Start_Index[idx])

        batch_paragraph = numpy.zeros((total_batch, p_length), dtype='f')
        batch_question = numpy.zeros((total_batch, q_length), dtype='f')
        batch_label = numpy.zeros((total_batch, 1), dtype='f')
        batch_label2 = numpy.zeros((total_batch, 2), dtype='f')

        token_index = self.get_dic_index('#END')
        for i in range(total_batch):
            for j in range(50):
                batch_paragraph[j] = token_index
                batch_question[j] = token_index

        batch_dump = numpy.zeros(shape=[total_batch], dtype=numpy.int32)

        index = 0

        for i in range(batch_size):
            idx = np_range[i]

            for j in range(self.WDP.Start_Index[idx], self.WDP.Stop_Index[idx]):
                batch_label[index, 0] = int(self.WDP.Labels[j])
                batch_label2[index, int(self.WDP.Labels[j])] = 1

                for k in range(len(self.WDP.Paragraphs[j])):
                    batch_paragraph[index, k] = self.get_dic_index(self.WDP.Paragraphs[j][k])
                for k in range(len(self.WDP.Questions[j])):
                    batch_question[index, k] = self.get_dic_index(self.WDP.Questions[j][k])

                index += 1

        for i in range(batch_size2):
            idx = np_range2[i]

            for j in range(self.RDP.Start_Index[idx], self.RDP.Stop_Index[idx]):
                batch_label[index, 0] = int(self.RDP.Labels[j])
                batch_label2[index, int(self.RDP.Labels[j])] = 1

                for k in range(len(self.RDP.Paragraphs[j])):
                    batch_paragraph[index, k] = self.get_dic_index(self.RDP.Paragraphs[j][k])
                for k in range(len(self.RDP.Questions[j])):
                    batch_question[index, k] = self.get_dic_index(self.RDP.Questions[j][k])

                index += 1

        return batch_paragraph, batch_question, batch_label, batch_label2, batch_dump

    def get_next_batch(self):
        batch_size = 0
        batch_size2 = 32

        p_length = 50
        q_length = 50

        np_range = numpy.arange(0, len(self.WDP.Start_Index) - 200, dtype='i')
        numpy.random.shuffle(np_range)

        np_range2 = numpy.arange(0, len(self.RDP.Start_Index) - 5000, dtype='i')
        numpy.random.shuffle(np_range2)

        paragraphs = []
        questions = []
        labels = []

        for i in range(batch_size):
            idx = np_range[i]

            for j in range(self.WDP.Start_Index[idx], self.WDP.Stop_Index[idx]):
                paragraphs.append(self.WDP.Paragraphs[j])
                questions.append(self.WDP.Questions[j])
                labels.append(self.WDP.Labels[j])

        for i in range(batch_size2):
            idx = np_range2[i]

            for j in range(self.RDP.Start_Index[idx], self.RDP.Stop_Index[idx]):
                paragraphs.append(self.RDP.Paragraphs[j])
                questions.append(self.RDP.Questions[j])
                labels.append(self.RDP.Labels[j])

        total_batch = 0

        for i in range(batch_size):
            idx = np_range[i]
            total_batch += (self.WDP.Stop_Index[idx] - self.WDP.Start_Index[idx])
        for i in range(batch_size2):
            idx = np_range2[i]
            total_batch += (self.RDP.Stop_Index[idx] - self.RDP.Start_Index[idx])

        batch_paragraph = numpy.zeros((total_batch, p_length, self.Word_Embedding_Dimension), dtype='f')
        batch_question = numpy.zeros((total_batch, q_length, self.Word_Embedding_Dimension), dtype='f')
        batch_label = numpy.zeros((total_batch, 1), dtype='f')
        batch_label2 = numpy.zeros((total_batch, 2), dtype='f')

        batch_dump = numpy.zeros(shape=[total_batch], dtype=numpy.int32)

        index = 0

        for i in range(batch_size):
            idx = np_range[i]

            for j in range(self.WDP.Start_Index[idx], self.WDP.Stop_Index[idx]):
                batch_label[index, 0] = int(self.WDP.Labels[j])
                batch_label2[index, int(self.WDP.Labels[j])] = 1

                batch_paragraph[index] = self.get_glove_sequence(length=p_length, tokens=self.WDP.Paragraphs[j])
                batch_question[index] = self.get_glove_sequence(length=q_length, tokens=self.WDP.Questions[j])

                index += 1

        for i in range(batch_size2):
            idx = np_range2[i]

            for j in range(self.RDP.Start_Index[idx], self.RDP.Stop_Index[idx]):
                batch_label[index, 0] = int(self.RDP.Labels[j])
                batch_label2[index, int(self.RDP.Labels[j])] = 1

                batch_paragraph[index] = self.get_glove_sequence(length=p_length, tokens=self.RDP.Paragraphs[j])
                batch_question[index] = self.get_glove_sequence(length=q_length, tokens=self.RDP.Questions[j])

                index += 1

        return batch_paragraph, batch_question, batch_label, batch_label2, batch_dump

    def get_test_Batch1(self):
        p_length = 50
        q_length = 50

        paragraphs = []
        questions = []
        labels = []

        idx = self.batch_index1

        for j in range(self.WDP.Start_Index[idx], self.WDP.Stop_Index[idx]):
            paragraphs.append(self.WDP.Paragraphs[j])
            questions.append(self.WDP.Questions[j])
            labels.append(self.WDP.Labels[j])

        total_batch = 0
        total_batch += (self.WDP.Stop_Index[idx] - self.WDP.Start_Index[idx])

        batch_paragraph = numpy.zeros((total_batch, p_length, self.Word_Embedding_Dimension), dtype='f')
        batch_question = numpy.zeros((total_batch, q_length, self.Word_Embedding_Dimension), dtype='f')
        batch_label = numpy.zeros((total_batch, 1), dtype='f')
        batch_label2 = numpy.zeros((total_batch, 2), dtype='f')

        index = 0

        for j in range(self.WDP.Start_Index[idx], self.WDP.Stop_Index[idx]):
            batch_label[index] = int(self.WDP.Labels[j])
            batch_label2[index, int(self.WDP.Labels[j])] = 1

            batch_paragraph[index] = self.get_glove_sequence(length=p_length, tokens=self.WDP.Paragraphs[j])
            batch_question[index] = self.get_glove_sequence(length=q_length, tokens=self.WDP.Questions[j])
            index += 1

        self.batch_index1 += 1

        return batch_paragraph, batch_question, batch_label, batch_label2, 0

    def get_test_Batch2(self):
        p_length = 50
        q_length = 50

        paragraphs = []
        questions = []
        labels = []

        idx = self.batch_index2

        for j in range(self.RDP.Start_Index[idx], self.RDP.Stop_Index[idx]):
            paragraphs.append(self.RDP.Paragraphs[j])
            questions.append(self.RDP.Questions[j])
            labels.append(self.RDP.Labels[j])

        total_batch = 0
        total_batch += (self.RDP.Stop_Index[idx] - self.RDP.Start_Index[idx])

        batch_paragraph = numpy.zeros((total_batch, p_length, self.Word_Embedding_Dimension), dtype='f')
        batch_question = numpy.zeros((total_batch, q_length, self.Word_Embedding_Dimension), dtype='f')
        batch_label = numpy.zeros((total_batch, 1), dtype='f')
        batch_label2 = numpy.zeros((total_batch, 2), dtype='f')

        index = 0

        for j in range(self.RDP.Start_Index[idx], self.RDP.Stop_Index[idx]):
            batch_label[index, 0] = int(self.RDP.Labels[j])
            batch_label2[index, int(self.RDP.Labels[j])] = 1

            batch_paragraph[index] = self.get_glove_sequence(length=p_length, tokens=self.RDP.Paragraphs[j])
            batch_question[index] = self.get_glove_sequence(length=q_length, tokens=self.RDP.Questions[j])
            index += 1

        self.batch_index2 += 1

        return batch_paragraph, batch_question, batch_label, batch_label2, 0