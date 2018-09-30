import numpy as np
from operator import itemgetter
import heapq
import collections
import hgtk

class Dictionary:
    def decompose_hangeul(self, string, max_length=20):
        # utf-8 hanguel start index: 44032, end index: 55203
        utf_start_hanguel = 44032
        utf_end_hanguel = 55203

        hanguel_index = 12593
        gap_index = 12643 - 12593

        index = 0

        result = ''

        for w in range(len(string)):
            if utf_start_hanguel <= ord(string[w]) <= utf_end_hanguel:
                letters = hgtk.letter.decompose(string[w])

                length = len(letters)

                for i in range(length):
                    if letters[i] != '':
                        result += letters[i]

            else:
                result += string[w]

        if len(result) > 1:
            return result[0:2]

        return result

    def __init__(self, do_indexing=True, do_post_processing=True):
        self.V_C_dictionary = []
        for i in range(ord('ㄱ'), ord('ㅎ') + 1):
            for j in range(ord('ㅏ'), ord('ㅣ') + 1):
                self.V_C_dictionary.append(chr(i) + chr(j))
        self.V_C_dictionary = np.array(self.V_C_dictionary, dtype='<U10')
        self.V_C_dictionary.sort()

        self.Dictionary = np.zeros(shape=[1])
        self.Inverted_Index = []
        self.Frequency = 0
        if do_post_processing is False and do_indexing is False:
            file = open('indexing/' + 'words', 'r', encoding='utf-8')
            TK = file.read().split('#')
            self.Dictionary = []
            for i in range(len(TK)):
                self.Dictionary.append(TK[i])
            file.close()
            self.Dictionary = np.array(self.Dictionary, dtype='<U30')

            file = open('indexing/' + 'titles', 'r', encoding='utf-8')
            TK = file.read().split('#')
            self.Wiki_Titles = []
            for i in range(len(TK)):
                self.Wiki_Titles.append(TK[i])
            file.close()
            self.Wiki_Titles = np.array(self.Wiki_Titles, dtype='<U30')

            file = open('indexing/' + 'config', 'r', encoding='utf-8')
            self.Documents_Num = int(file.read())
            file.close()

        else:
            #Dictionary를 만들 데이터 파일의 경로
            path = 'doc_file'

            file = open(path, 'r', encoding='utf-8')
            TK = file.read().replace('\a', ' ').replace('  ', ' ').split()
            print(len(TK))
            print('text file read')
            self.make_dictionary(TK)

            file.close()
            file = open(path, 'r', encoding='utf-8')

            self.Wiki_Titles = []
            documents = file.read().split('\a')
            for i in range(len(documents)):
                self.Wiki_Titles.append(documents[i].split('\n')[0])
            self.Wiki_Titles = np.array(self.Wiki_Titles, dtype='<U30')

            self.Documents_Num = len(documents)
            self.fragment_size = 30000
            self.Store_tile = 1000
            if do_indexing is True:
                self.indexing(documents)
            if do_post_processing is True:
                self.post_processing()

    def indexing(self, documents):
        files = []
        file_indexes = []

        for i in range(ord('ㄱ'), ord('ㅎ') + 1):
            for j in range(ord('ㅏ'), ord('ㅣ') + 1):
                file = open('indexing/' + chr(i) + chr(j) + '_indexing_file', 'w', encoding='utf-8')
                files.append(file)
                file_indexes.append(chr(i) + chr(j))
        file = open('indexing/' + 'english' + '_indexing_file', 'w', encoding='utf-8')
        files.append(file)
        file_indexes = np.array(file_indexes, dtype='<U20')
        arg_file_indexes = np.argsort(file_indexes)
        file_indexes.sort()

        for i in range(len(documents)):
            TK = str(documents[i]).replace('\n', ' ').replace('  ', ' ').split()
            list = self.least_common_values(TK)
            for j in range(len(list)):
                idx = file_indexes.searchsorted(self.decompose_hangeul(list[j][0][0]))
                if idx < arg_file_indexes.shape[0]:
                    if file_indexes[idx] != self.decompose_hangeul(list[j][0][0]):
                        idx = len(files) - 1
                    else:
                        idx = arg_file_indexes[idx]
                else:
                    idx = len(files) - 1

                files[idx].write(list[j][0] + ',' + '{}'.format(i) + ',' + str(list[j][1]) + '\n')
                #print(file_indexes[idx])
                #print(list[j][0] + ',' + '{}'.format(i) + ',' + str(list[j][1]) + '\n')
                #print()
                #input()

            del list
            print(i, '/', len(documents))
        for i in range(len(files)):
            files[i].close()

    def post_processing(self):
        #post processing
        file = open('indexing/' + 'config', 'w', encoding='utf-8')
        file.write(str(self.Documents_Num))
        file.close()

        file = open('indexing/' + 'titles', 'w', encoding='utf-8')
        for i in range(self.Wiki_Titles.shape[0]):
            file.write(self.Wiki_Titles[i] + '#')
        file.write(self.Wiki_Titles[self.Wiki_Titles.shape[0] - 1])
        file.close()

        file = open('indexing/' + 'words', 'w', encoding='utf-8')
        for i in range(self.Dictionary.shape[0]):
            file.write(self.Dictionary[i] + '#')
        file.write(self.Dictionary[self.Dictionary.shape[0] - 1])
        file.close()

        ###########################################
        r_words = []

        file_ = open('indexing/' + 'english' + '_indexing_file', 'r', encoding='utf-8')
        file = open('indexing/' + 'result' + 'english' + '_indexing_file', 'w', encoding='utf-8')

        lines = file_.read()
        if len(lines) > 0:
            lines = lines[0:len(lines) - 1].split('\n')

            arr_lines = np.array(lines, dtype='<U250')
            arr_lines.sort()

            TK = arr_lines[0].split(',')
            file.write(TK[1] + '!,' + TK[2])
            last_word = TK[0]

            r_words.append(TK[0])

            result = ''

            for k in range(1, arr_lines.shape[0] - 1):
                TK = arr_lines[k].split(',')
                if TK[0] == last_word:
                    result += '!#' + TK[1] + '!,' + TK[2]
                else:
                    result += '\n' + TK[1] + '!,' + TK[2]
                    r_words.append(TK[0])
                last_word = TK[0]

            file.write(result)

        file.write('\n')
        result = ''
        for k in range(len(r_words) - 1):
            result += r_words[k] + '#'
        if len(r_words) > 0:
            file.write(result)
            file.write(r_words[len(r_words) - 1])
        file.close()
        #######################################

        for i in range(ord('ㄱ'), ord('ㅎ') + 1):
            for j in range(ord('ㅏ'), ord('ㅣ') + 1):
                r_words = []

                file_ = open('indexing/' + chr(i) + chr(j) + '_indexing_file', 'r', encoding='utf-8')
                file = open('indexing/' + 'result' + chr(i) + chr(j) + '_indexing_file', 'w', encoding='utf-8')

                lines = file_.read()
                if len(lines) > 0:
                    lines = lines[0:len(lines) - 1].split('\n')

                    arr_lines = np.array(lines, dtype='<U250')
                    arr_lines.sort()

                    TK = arr_lines[0].split(',')
                    file.write(TK[1] + '!,' + TK[2])
                    last_word = TK[0]

                    r_words.append(TK[0])

                    result = ''

                    for k in range(1, arr_lines.shape[0] - 1):
                        TK = arr_lines[k].split(',')
                        if TK[0] == last_word:
                            result += '!#' + TK[1] + '!,' + TK[2]
                        else:
                            result += '\n' + TK[1] + '!,' + TK[2]
                            r_words.append(TK[0])
                        last_word = TK[0]

                    file.write(result)

                file.write('\n')
                result = ''
                for k in range(len(r_words) - 1):
                    result += r_words[k] + '#'
                if len(r_words) > 0:
                    file.write(result)
                    file.write(r_words[len(r_words) - 1])
                file.close()

    def least_common_values(self, array, to_find=None):
        counter = collections.Counter(array)
        if to_find is None:
            return sorted(counter.items(), key=itemgetter(1), reverse=False)
        return heapq.nsmallest(to_find, counter.items(), key=itemgetter(1))

    def make_dictionary(self, words):
        data_list = self.least_common_values(words)
        print(data_list[0])

        num_word = 0

        print('num word:', num_word)
        self.Dictionary = np.zeros(shape=[len(data_list)], dtype='<U20')

        index = 0

        for i in range(len(data_list)):
            self.Dictionary[index] = data_list[i][0]
            index += 1
        print('index:', index)
        self.Dictionary.sort()

    def get_dictionary_index(self, word):
        v_c = self.decompose_hangeul(word[0])
        idx = self.V_C_dictionary.shape[0]
        for i in range(self.V_C_dictionary.shape[0]):
            #print(v_c, self.V_C_dictionary[i])
            #input()
            if v_c == self.V_C_dictionary[i]:
                idx = i
                break
        if idx == self.V_C_dictionary.shape[0]:
            v_c = 'english'

        idx = self.Dictionary.searchsorted(word)
        if idx == self.Dictionary.shape[0]:
            return -1, v_c
        if self.Dictionary[idx] == word:
            return idx, v_c
        else:
            return -1, v_c
