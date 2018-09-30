import Dictionary
import numpy as np

class Extension_memory:
    def __init__(self, size):
        self.Time_Index = 0
        #LRU를 실행하기 위한 매개변수, 특정 구간에서 전부 초기화를 하는데에도 사용

        self.Size = size
        self.Start_Index = np.zeros(shape=[size], dtype=np.int32)
        self.Stop_Index = np.zeros(shape=[size], dtype=np.int32)
        self.Time = np.zeros(shape=[size], dtype=np.int32)
        self.Dictionary_Index = np.zeros(shape=[size], dtype=np.int32)
        self.Memory = np.zeros(shape=[size * 10000 * 2], dtype=np.int32)
        self.Memory_ = np.zeros(shape=[size * 10000 * 2], dtype=np.int32)

        for i in range(size):
            self.Dictionary_Index[i] = -1
            self.Time[i] = -1

        for i in range(size * 10000 * 2):
            self.Memory[i] = -1
            self.Memory_[i] = -1

    def refresh_Memory(self):
        for i in range(self.Size):
            self.Dictionary_Index[i] = -1
            self.Time[i] = -1

        for i in range(self.Size * 10000 * 2):
            self.Memory[i] = -1
            self.Memory_[i] = -1

    def check_Memory(self, index):
        for i in range(self.Size):
            if index == self.Dictionary_Index[i]:
                self.Time_Index += 1
                self.Time[i] = self.Time_Index

                return True

        return False

    def register_Memory(self, word, line):
        TK = line.split('!#')
        length = len(TK)
        seq_count = 0

        dic_idx = -1

        is_okay = False

        for i in range(self.Size):
            if self.Dictionary_Index[i] == -1:
                dic_idx = i
                break
        if dic_idx != -1:
            for i in range(self.Size * 10000 * 2):
                #find place with first fit
                if self.Memory[i] == -1:
                    seq_count += 1
                else:
                    seq_count = 0

                if seq_count == length:
                    self.Time[dic_idx] = self.Time_Index
                    self.Start_Index[dic_idx] = i - length
                    self.Stop_Index[dic_idx] = i + 1
                    self.Dictionary_Index[dic_idx] = word

                    for j2, j in enumerate(range(i - length + 1, i + 1)):
                        try:
                            self.Memory[j] = int(TK[j2].split('!,')[0])
                            self.Memory_[j] = int(TK[j2].split('!,')[1])
                        except:
                            self.Memory[j] = int(TK[j2].split(',')[1])
                            self.Memory_[j] = 1
                    is_okay = True
                    break

        if is_okay is False:
            #replacement with LRU polisy
            min = 9999999
            min_index = -1
            for i in range(self.Size):
                if min > self.Time[i]:
                    min = self.Time[i]
                    min_index = i

            for i in range(self.Start_Index[min_index], self.Stop_Index[min_index]):
                self.Memory[i] = -1
                self.Memory_[i] = -1
            self.Dictionary_Index[min_index] = -1

            #공간이 완전히 생길때까지 반복
            self.register_Memory(word, line)

    def get_data(self, index):
        for i in range(self.Memory.shape[0]):
            if index == self.Dictionary_Index[i]:
                data_size = self.Stop_Index[i] - self.Start_Index[i]
                memory = np.zeros(shape=[data_size], dtype=np.int32)
                memory_ = np.zeros(shape=[data_size], dtype=np.int32)
                for j in range(data_size):
                    memory[j] = self.Memory[j + self.Start_Index[i]]
                    memory_[j] = self.Memory_[j + self.Start_Index[i]]

                if self.Time_Index == 100000:
                    self.refresh_Memory()

                return memory, memory_


class Agent:
    def __init__(self, do_post_processing=False, do_indexing=False):
        self.Data_holder = Dictionary.Dictionary(do_post_processing=do_post_processing, do_indexing=do_indexing)
        self.Extension_Memory = Extension_memory(3000)

        self.Frequency = np.zeros(shape=[self.Data_holder.Documents_Num], dtype=np.float32)
        self.Num_Words = np.zeros(shape=[self.Data_holder.Documents_Num], dtype=np.float32)

        print('start!')

    def search(self, texts):
        print('search!')

        for i in range(len(texts)):
            print(i, '/', len(texts))

            idx, v_c = self.Data_holder.get_dictionary_index(texts[i])
            if idx != -1:
                if self.Extension_Memory.check_Memory(idx) is True:
                    m1, m2 = self.Extension_Memory.get_data(idx)

                    for j in range(m1.shape[0]):
                        self.Num_Words[m1[j]] += 1
                        self.Frequency[m1[j]] += m2[j]
                else:
                    file = open('result' + v_c + '_indexing_file', 'r', encoding='utf-8')
                    lines = file.read().split('\n')
                    dictionary = lines[len(lines) - 1].split('#')
                    dictionary = np.array(dictionary, dtype='<U50')

                    index = 0
                    for j in range(dictionary.shape[0]):
                        if dictionary[j] == texts[i]:
                            index = j
                            break

                    TK = lines[index].split('!#')
                    print(texts[i], ',', v_c)
                    #input()
                    for j in range(len(TK)):
                        try:
                            doc_number = int(TK[j].split('!,')[0])
                            freq = int(TK[j].split('!,')[1])
                        except:
                            doc_number = int(TK[j].split('!,')[1])
                            freq = 1
                        self.Num_Words[doc_number] += 1
                        self.Frequency[doc_number] += freq / len(TK)

                    self.Extension_Memory.register_Memory(idx, lines[index])

        max_score = -999
        max_index = -1
        for i in range(self.Data_holder.Documents_Num):
            if self.Num_Words[i] > 0:
                if self.Frequency[i] / self.Num_Words[i] > max_score:
                    max_score = self.Frequency[i] / self.Num_Words[i]
                    max_index = i
        print(max_score)
        return max_index










