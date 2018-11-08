"""
author: Cho Sanghyeon(Pusan National Univ.)
email: delosycho@gmail.com

빈도수를 기반으로 하는 추천 알고리즘 클래스

-함수 목록-
make_dictionary: 빈도수 계산에 필요한 사전을 만듭니다
word2idx: 해당 단어에 해당하는 사전에서의 인덱스로 출력합니다
register_word: 사전을 기반으로 빈도수를 계산합니다
classification: 빈도수를 기반으로 추천/비추천의 여부를 계산하고 결정합니다(tf-idf나 naive bayes중 하나를 선택해야함)
update_dictionary: 새로운 추가 텍스트가 있을 경우, 사전 정보를 업데이트 합니다
---------
"""

import numpy as np
import math

class Classifier:
    def __init__(self, number_of_class):
        self.dictionary = []
        #전체 단어사전, 이 사전의 index를 통해 단어의 빈도수에 접근

        self.Frequency = np.zeros(shape=[0, 0], dtype=np.int32)
        self.number_of_class = number_of_class

    def make_dictionary(self, words):
        for i in range(len(words)):
            try:
                self.dictionary.index(words[i])
            except:
                self.dictionary.append(words[i])

        self.Frequency = np.zeros(shape=[self.number_of_class, len(self.dictionary)], dtype=np.int32)

    def update_dictionary(self, words, class_number):
        for i in range(len(words)):
            try:
                self.dictionary.index(words[i])
            except:
                self.dictionary.append(words[i])

        new_frequency = np.zeros(shape=[self.number_of_class, len(self.dictionary)], dtype=np.int32)
        for i in range(self.number_of_class):
            for j in range(self.Frequency.shape[0]):
                new_frequency[i, j] = self.Frequency[i, j]

        for i in range(len(words)):
            index = self.word2idx(words[i])
            if index != -1:
                new_frequency[class_number, index] += 1

        self.Frequency = new_frequency

    def word2idx(self, word):
        try:
            index = self.dictionary.index(word)
        except:
            index = -1

        return index

    def register_word(self, class_number, words):
        for i in range(len(words)):
            index = self.word2idx(words[i])
            if index != -1:
                self.Frequency[class_number, index] += 1

    #tf-idf
    def classification_tf_idf(self, words):
        max_score = -9999
        max_class = -1

        for k in range(self.number_of_class):
            #print(self.Frequency[k])

            total_score = 0
            num = 0

            for i in range(len(words)):
                index = self.word2idx(words[i])
                if index != -1:
                    if self.Frequency[k, index] > 0:
                        tf = 0.5 + 0.5 * (self.Frequency[k, index] / np.max(self.Frequency))

                        count = 1
                        for q in range(self.number_of_class):
                            idx_ = self.word2idx(words[i])
                            if idx_ != -1:
                                if self.Frequency[q, idx_] > 0:
                                    count += 1

                        idf = math.log(count)
                        tf_idf = tf * idf

                        num += 1
                        total_score += tf_idf

            if max_score < total_score:
                max_score = total_score
                max_class = k
            print(total_score)
        return max_class

        # tf-idf
    def classification_tf_idf_with_score(self, words):
        max_score = -9999
        max_class = -1

        scores = []

        for k in range(self.number_of_class):
            #print(self.Frequency[k])

            total_score = 0
            num = 0

            for i in range(len(words)):
                index = self.word2idx(words[i])
                if index != -1:
                    if self.Frequency[k, index] > 0:
                        tf = 0.5 + 0.5 * (self.Frequency[k, index] / np.max(self.Frequency))

                        count = 1
                        for q in range(self.number_of_class):
                            idx_ = self.word2idx(words[i])
                            if idx_ != -1:
                                if self.Frequency[q, idx_] > 0:
                                    count += 1

                        idf = math.log(count)
                        tf_idf = tf * idf

                        num += 1
                        total_score += tf_idf

            if max_score < total_score:
                max_score = total_score
                max_class = k
            scores.append(total_score)

        score = scores[0] - scores[1]

        return score

    def idf_with_score(self, words, idx):
        max_score = -9999
        max_class = -1

        scores = []

        for k in range(self.number_of_class):
            #print(self.Frequency[k])

            total_score = 0
            num = 0

            for i in range(len(words)):
                index = self.word2idx(words[i])
                if index != -1:
                    if self.Frequency[k, index] > 0:

                        count = 1
                        for q in range(self.number_of_class):
                            idx_ = self.word2idx(words[i])
                            if idx_ != -1:
                                if self.Frequency[q, idx_] > 0:
                                    count += 1

                        idf = math.log(count)

                        num += 1
                        total_score += idf

            if max_score < total_score:
                max_score = total_score
                max_class = k
            scores.append(total_score)

        return scores[idx]

    #naive bayes
    def classification_naive_bayes(self, words):
        max_prob = -9999
        max_class = -1

        for k in range(self.number_of_class):
            total_probability = 0

            for i in range(len(words)):
                index = self.word2idx(words[i])
                if index != -1:
                    if self.Frequency[k, index] > 0:
                        prob = self.Frequency[k, index] / np.sum(self.Frequency)
                        total_probability += math.log(prob)

            if max_prob < total_probability:
                max_prob = total_probability
                max_class = k

        return max_class

