import numpy

class Dataholder:
    def __init__(self):
        label_file = open('label_file', 'r', encoding='utf-8')
        question_file = open('question_file', 'r', encoding='utf-8')
        sentence_file = open('sentence_file', 'r', encoding='utf-8')
        label_index_file = open('label_index_file', 'r', encoding='utf-8')

        self.Labels = label_file.read().split('\n')
        #print(self.Labels)
        #input()
        self.Labels_index = label_index_file.read().split('\n')
        self.Questions = question_file.read().split('\n')
        self.Sentences = sentence_file.read().split('\n')
        self.Test_Batch_Index = 0

        print(len(self.Questions), len(self.Sentences), len(self.Labels))
#        input()

        self.Word_Embedding_Dimension = 100
        self.Test_Batch_index = 0

    def QA_Test_Batch(self):
        idx = self.Test_Batch_index
        try:
            labels_start = int(self.Labels_index[idx].split('#')[0])
        except:
            labels_start = 0
        labels_stop = int(self.Labels_index[idx].split('#')[1])

        TK1 = self.Sentences[idx].split('\t')
        TK2 = self.Labels[idx].split('\t')

        cur_batch_sen = []
        cur_batch_que = []
        cur_batch_lab = []

        for aa in range(len(TK1) - 1):
            if len(TK1[aa].split()) > 2:
                try:
                    cur_batch_lab.append(int(TK2[aa]))
                    cur_batch_sen.append(TK1[aa].strip())
                    cur_batch_que.append(self.Questions[idx].strip())
                except:
                    print(labels_start, labels_stop, idx)
                    print(len(self.Sentences), len(TK2), aa)
                    print(len(self.Questions), len(TK1))
                    print(TK1)
                    print('eeror')
                    #input()

        #batch_sentences = numpy.zeros((cur_batch, cur_length, self.Word_Embedding_Dimension), dtype='f')
        #batch_questions = numpy.zeros((cur_batch, cur_length, self.Word_Embedding_Dimension), dtype='f')
        #batch_label = numpy.zeros((cur_batch, 2), dtype='i')

        self.Test_Batch_index += 1

        return cur_batch_sen, cur_batch_que, cur_batch_lab

label_index_file = open('label_index_file__', 'w', encoding='utf-8')
label_file = open('label_file__', 'w', encoding='utf-8')
question_file = open('question_file__', 'w', encoding='utf-8')
sentence_file = open('sentence_file__', 'w', encoding='utf-8')

label_index = 0

data = Dataholder()
for i in range(len(data.Labels) - 10):
    batch_sentences, batch_questions, batch_label = data.QA_Test_Batch()
    label_index_file.write(str(label_index))

    for j in range(len(batch_sentences) - 1):
        sentence_file.write(batch_sentences[j])
        sentence_file.write('\n')
        question_file.write(batch_questions[j])
        question_file.write('\n')
        label_file.write(str(batch_label[j]))
        label_file.write('\n')

    label_index += len(batch_sentences)
    label_index_file.write(' ')
    label_index_file.write(str(label_index))
    label_index_file.write('\n')

label_file.close()
label_index_file.close()
question_file.close()
sentence_file.close()

label_index_file = open('label_index_file__', 'r', encoding='utf-8')
label_file = open('label_file__', 'r', encoding='utf-8')
question_file = open('question_file__', 'r', encoding='utf-8')
sentence_file = open('sentence_file__', 'r', encoding='utf-8')

print(len(sentence_file.read().split('\n')))
print(len(question_file.read().split('\n')))
print(len(label_file.read().split('\n')))

