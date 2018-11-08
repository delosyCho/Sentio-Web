import Recommend_Agent
import Frequency_Based_Recommend
from utils import *

class Agent:
    def __init__(self, number_of_class):
        print('형태소 분석기를 실행하세요')
        self.st = Stemmer()

        self.use_buffer = True  # 버퍼 시스템 사용여부
        self.limit_buffer = 10  # 버퍼 업데이트 시점 결정(데이터의 쌓인 갯수)

        # dataset
        self.data = None
        self.buffer_data = []
        self.buffer_label = []

        self.recommend_agent_Frequency = Frequency_Based_Recommend.Classifier(8)

        total_text = ''
        train_texts = []

        for i in range(5):
            file_texts = open('newdata_' + str(i) + '.txt', 'r', encoding='utf-8')
            texts = file_texts.read()
            texts, POS = self.st.stemming(texts)
            total_text += texts
            train_texts.append(texts)

        total_text = total_text.split()
        self.recommend_agent_Frequency.make_dictionary(total_text)

        for i in range(5):
            self.recommend_agent_Frequency.register_word(i, train_texts[i])

    def add_buffer(self, data, class_number):
        # 시스템에 의해서 사용자의 새로운 데이터가 생성되었을때 저장하기 위한 함수
        # add buffer를 사용하는 이유는 매번 전체 데이터를 갱신하는 것은 시스템의 전체 속도에 지장을 줄 수 있음
        self.buffer_data.append(data)
        self.buffer_label.append(class_number)

    def update_data(self, model_type):
        # model_type은 recommend_agent의 type을 의미함, 0부터 (frequency, similarity)의 순서
        if model_type == 0:
            self.recommend_agent_Frequency.update_dictionary(self.buffer_data, self.buffer_label)
        elif model_type == 1:
            self.recommend_agent_Similarity.update_data(self.buffer_data, self.buffer_label)