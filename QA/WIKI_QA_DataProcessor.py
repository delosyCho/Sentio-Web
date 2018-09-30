import numpy
import codecs

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

        path = 'C:/Users/Administrator/Desktop/qadataset/WikiQA.tsv'
        index = 0

        self.questions_ = []
        self.one_paragraph = []
        self.zero_paragraph = []

        self.Questions = []
        self.Paragraphs = []
        self.Labels = []
        self.Start_Index = []
        self.Stop_Index = []

        question = []
        parapgraphs = []
        answer = -1
        last_index = 0

        file = open(path, 'r', encoding='utf8')
        lines = file.read().split('\n')

        for i in range(1, len(lines) - 1):
            Tokens = lines[i].split('	')

            current_Index = int(Tokens[0][1:len(Tokens[0])])
            if current_Index != last_index:
                if answer != -1:
                    self.questions_.append(question)
                    self.one_paragraph.append(answer)
                    self.zero_paragraph.append(parapgraphs)

                question = []
                parapgraphs = []
                answer = -1
                #print(Tokens)
                #input()

            #print(current_Index)
            if Tokens[6] == '1':
                question = Tokens[1]
                answer = Tokens[5]
            else:
                parapgraphs.append(Tokens[5])

            last_index = current_Index

            #input()

        print(len(self.one_paragraph))
        print(len(self.one_paragraph))

        count = 0
        self.Start_Index.append(count)

        for i in range(len(self.one_paragraph)):
            for j in range(len(self.zero_paragraph[i])):
                self.Paragraphs.append(self.zero_paragraph[i][j].split())
                self.Questions.append(self.questions_[i].split())
                self.Labels.append(0)
                count += 1

            self.Paragraphs.append(self.one_paragraph[i].split())
            self.Questions.append(self.questions_[i].split())
            self.Labels.append(1)
            count += 1

            self.Start_Index.append(count)
            self.Stop_Index.append(count)

        self.Stop_Index.append(count)

        self.Batch_Index = len(self.Questions)
        print(len(self.Questions))
        print(len(self.Start_Index))
        print(len(self.Stop_Index))

        print(self.Start_Index[len(self.Start_Index) - 1])

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

    def get_sequence_batch(self):
        cont = True

        cur_batch = 80
        if self.Batch_Index + cur_batch > len(self.Start_Index):
            cur_batch = len(self.Start_Index) - self.Batch_Index
            cont = False
        cur_length = 70
        q_max = 50

        total_batch = 0

        np_range = numpy.arange(0, len(self.Start_Index), dtype='i')
        numpy.random.shuffle(np_range)

        for i in range(self.Batch_Index, self.Batch_Index + cur_batch):
            idx = i
            total_batch += (self.Stop_Index[idx] - self.Start_Index[idx])

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

    def get_test_batch(self):
        cur_batch = 1
        cur_length = 70
        q_max = 50

        total_batch = 0

        np_range = numpy.arange(0, len(self.Start_Index), dtype='i')
        numpy.random.shuffle(np_range)

        idx = np_range[0]
        total_batch += (self.Stop_Index[idx] - self.Start_Index[idx])

        batch_paragraph = numpy.zeros((total_batch, cur_length, self.Word_Embedding_Dimension), dtype='f')
        batch_question = numpy.zeros((total_batch, q_max, self.Word_Embedding_Dimension), dtype='f')
        batch_label = numpy.zeros((total_batch, 2), dtype='f')

        index = 0

        idx = np_range[0]

        for j in range(self.Start_Index[idx], self.Stop_Index[idx]):
            batch_label[index, int(self.Labels[j])] = 1
            batch_paragraph[index] = self.get_glove_sequence(length=cur_length, tokens=self.Paragraphs[j])
            batch_question[index] = self.get_glove_sequence(length=q_max, tokens=self.Questions[j])
            index += 1

        return batch_paragraph, batch_question, batch_label, idx, idx