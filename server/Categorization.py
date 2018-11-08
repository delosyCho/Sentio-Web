"""
author: Cho Sanghyeon(Pusan National Univ.)
email: delosycho@gmail.com

사용자가 사용하고 있는 문서들의 카테고리를 분석하고, 분석 된 카테고리를 기반으로 추천서비스를 제공하기 위한
알고리즘과 함수들이 포함 된 클래스입니다

-함수 목록-

---------
"""
import Data
import Frequency_Based_Recommend
import Similarity_Based_Recommend
import NN_Based_Recommend


class Categorization:
    def __init__(self):
        """
        NN_Based Model은 Scheduler를 통해 작동하기 때문에 유일하기 다른 매커니즘을 거쳐서 작동합니다
        """

        """
        카테고리 부가 정보
       """
        category_dic = []

        self.Number_of_class = 2
        self.Model_Code = 0     #사용 할 모델
        self.Function_Code = 0  #모델 내에서 사용 할 분류기의 함수 번호(소스코드 내의 순서대로)

        self.classification_agent_Frequency = None
        self.classification_agent_Similarity = None
        self.classification_agent_NN = None

        """
        사용하길 원하는 데이터셋이 있다면 이 부분에서 데이터셋의 경로 변경을 하면 됩니다
        (이 클래스에서는 추천 모델에서와 다르게 유동적인 데이터를 사용하지 않으며, 기본데이터를 제공합니다)
        """
        data_path = ''
        label_path = ''
        if self.Model_Code == 0:
            file = open(data_path, 'r', encoding='utf-8')
            words = file.read().split('#')

            file_ = open(label_path, 'r', encoding='utf-8')
            labels = file_.read().split('#')

            self.classification_agent_Frequency = Frequency_Based_Recommend.Classifier(self.Number_of_class)
            self.classification_agent_Frequency.make_dictionary(words=words)
            self.classification_agent_Frequency.register_word(class_number=labels, words=words)
        elif self.Model_Code == 1:
            data_holder = Data.DataHolder(path=data_path, name='data_sim')

            file_ = open(label_path, 'r', encoding='utf-8')
            labels = file_.read().split('#')

            self.classification_agent_Similarity = Similarity_Based_Recommend.Classifier(batch_size=data_holder.data.shape[0],
                                                                                         number_of_class=self.Number_of_class,
                                                                                         dimension=data_holder.data.shape[1],
                                                                                         data=data_holder.data,
                                                                                         label=labels,
                                                                                         dtype=data_holder.dtype)
        ###########################

    def get_category(self, data):
        category_idx = -1
        if self.Model_Code == 0:
            if self.Function_Code == 0:
                category_idx = self.classification_agent_Frequency.classification_tf_idf(data)
            elif self.Function_Code == 1:
                category_idx = self.classification_agent_Frequency.classification_naive_bayes(data)
        elif self.Model_Code == 1:
            if self.Function_Code == 0:
                category_idx = self.classification_agent_Similarity.knn_classifier(data, k=5)
                #k 값에 따라서 결과가 달라질 수 있음
            elif self.Function_Code == 1:
                category_idx = self.classification_agent_Similarity.svm_classifier(data)

        return category_idx