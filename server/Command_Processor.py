import numpy as np
import socket
import datetime
from html_utils import *
from Agent import *
import dDANN
import Recommend_Agent
"""

"""

class Command_Processor:
    def __init__(self):
        """
        규칙 리스트와 기능 설명은 개발자 문서를 참고
        """
        try:
            recommend_file = open('recommend.txt', 'r', encoding='utf-8')
        except:
            recommend_file = open('recommend.txt', 'w', encoding='utf-8')
            recommend_file.write('word')
            recommend_file.close()
            recommend_file = open('recommend.txt', 'r', encoding='utf-8')
        try:
            non_recommend_file = open('non_recommend.txt', 'r', encoding='utf-8')
        except:
            non_recommend_file = open('recommend.txt', 'w', encoding='utf-8')
            non_recommend_file.write('word')
            non_recommend_file.close()
            non_recommend_file = open('recommend.txt', 'r', encoding='utf-8')
        recommend_texts = recommend_file.read().split()
        non_recommend_texts = non_recommend_file.read().split()

        total_texts = []
        for i in range(len(recommend_texts)):
            total_texts.append(recommend_texts[i])
        for i in range(len(non_recommend_texts)):
            total_texts.append(non_recommend_texts[i])

        self.recommend_agent = Recommend_Agent.Agent()
        self.recommend_agent.recommend_agent_Frequency.make_dictionary(total_texts)
        self.recommend_agent.recommend_agent_Frequency.register_word(0, recommend_texts)
        self.recommend_agent.recommend_agent_Frequency.register_word(1, recommend_texts)

        recommend_file.close()
        non_recommend_file.close()

        self.number_dic = ['영', '일', '이', '삼', '사', '오', '육', '칠', '팔', '구', '십', '십일', '십이', '십삼',
                           '십사', '십오']
        self.category_dic = ['정치', '경제', '사회', '아이티', '생활', '날씨', '게임', '스포츠', '연예', '소설', '만화', '애니메이션',
                             '프로그래밍', '입시', '음악', '영화', '음식', '자동차']

        self.recommend_direction = False
        self.arg_index = 0

        self.word_dic = []
        self.word_values = []

        self.Href_Idx = 0

        self.Command_Dic = ['이동', '설정', '다음글', '검색', '목록', '다음', '선택', '리스트', '재생', '추천', '비추천',
                            '탐색', '메인', '추천']
        self.Function_list = []

        self.Function_list.append(self.Function_1)
        self.Function_list.append(self.Function_2)
        self.Function_list.append(self.Function_3)
        self.Function_list.append(self.Function_4)
        self.Function_list.append(self.Function_5)
        self.Function_list.append(self.Function_6)
        self.Function_list.append(self.Function_7)
        self.Function_list.append(self.Function_8)
        self.Function_list.append(self.Function_9)
        self.Function_list.append(self.Function_10)
        self.Function_list.append(self.Function_11)
        self.Function_list.append(self.Function_12)
        self.Function_list.append(self.Function_13)
        self.Function_list.append(self.Function_14)

        self.Url = ''
        self.Complete_Url = ''
        self.current_text = ''
        self.Category = 0

        self.Rule = ''
        self.Rule_List = np.zeros(shape=[1], dtype='<U20')
        self.Rule_Values = np.zeros(shape=[1], dtype='<U20')

        """
        data file load
        """
        data_file = open('data_file', 'r', encoding='utf-8')
        lines = data_file.read().replace('﻿', '').split('\n')

        self.page_urls = []
        self.page_names = []

        self.page_urls_c = []
        self.body_tags = []
        self.body_IDs = []

        self.page_urls_a = []
        self.href_tags = []
        self.href_IDs = []

        self.page_urls_r = []
        self.rules = []
        self.rules_value = []

        self.page_urls_p = []
        self.pref_tags = []
        self.pref_IDs = []
        self.pref_names = []

        for i in range(len(lines)):
            try:
                if len(lines[i]) > 1:
                    TK = lines[i].split('#')
                    if int(TK[0]) == 0:
                        self.page_urls_c.append(TK[1])
                        self.body_tags.append(TK[2])
                        self.body_IDs.append(TK[3])
                    elif int(TK[0]) == 1:
                        self.page_urls_a.append(TK[1])
                        self.href_tags.append(TK[2])
                        self.href_IDs.append(TK[3])
                    elif int(TK[0]) == 2:
                        self.page_urls.append(TK[1])
                        self.page_names.append(TK[2])
                    elif int(TK[0]) == 3:
                        self.page_urls_r.append(TK[1])
                        self.rules.append(TK[2])
                        self.rules_value.append(TK[3])
                    elif int(TK[0]) == 4:
                        self.word_dic.append(TK[1])
                        self.word_values.append(TK[2])
                    elif int(TK[0]) == 5:
                        self.page_urls_p.append(TK[1])
                        self.pref_tags.append(TK[3])
                        self.pref_IDs.append(TK[4])
                        self.pref_names.append(TK[2])
            except Exception as e:
                print(e)
                print(lines[i])
                input()
        #crawler
        self.Crawler = Web_Crawler()

        #self.agent = Agent(do_indexing=False, do_post_processing=False)
        try:
            date_log = open('date_log', 'r', encoding='utf-8')
            time = str(datetime.datetime.now()).split(' ')[0]
            time2 = date_log.read().split(' ')[0]
            print(time, time2)
            if time != time2:
                print('a')
                self.crawl_base_document()
                self.agent = Agent(do_indexing=True, do_post_processing=True)
            else:
                self.agent = Agent(do_indexing=False, do_post_processing=False)
        except:
            self.crawl_base_document()
            self.agent = Agent(do_indexing=True, do_post_processing=True)

        """
        #####형태소 분석기, 딥러닝 모델##########
        HOST = ''  # 호스트를 지정하지 않으면 가능한 모든 인터페이스를 의미한다.
        PORT = 7979  # 포트지정
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((HOST, PORT))
        s.listen(1)  # 접속이 있을때까지 기다림
        self.conn, addr = s.accept()  # 접속 승인

        self.classifier = dDANN.One_Model()
        # classifier = dDANN.One_Model()
        #####
        """
    def url_check(self, url):
        dics = ['.kr', '.com', '.net', '.tv']
        for dic in dics:
            idx = url.find(dic)
            if idx != -1:
                return True
        return False

    def crawl_with_href(self):
        scores = np.zeros(shape=[len(self.Crawler.Href_link_list)], dtype=np.int32)

        for i in range(self.Crawler.Href_link_list):
            Link = str(self.Crawler.Href_link_list[i]).replace('amp;', '')

            if self.url_check(Link) is True:
                Url = Link
            else:
                if Link[0] == '/':
                    Url = self.Url + Link
                else:
                    Url = self.Url + '/' + Link

            idx = self.search_page_c(self.Url)
            if idx != -1:
                body_tag = self.body_tags[idx]
                body_id = self.body_IDs[idx]

                print(Url, body_tag, body_id)
                # print(self.body_IDs)
                # print(self.body_tags)
                # input()

                body_text = self.Crawler.get_article_text(url=Url, tag1='div', tag2=body_tag,
                                                          tag3=body_id)
                sc = self.recommend_agent.recommend_agent_Frequency.classification_tf_idf_with_score(body_text.split())
                scores[i] = sc

        self.arg_index = scores.argsort()

    def crawl_base_document(self):
        date_log = open('date_log', 'w', encoding='utf-8')
        date_log.write(str(datetime.datetime.now()))
        date_log.close()

        doc_file = open('doc_file', 'w', encoding='utf-8')

        for i in range(0, len(self.page_urls_p)):
            self.Crawler.Href_title_list = []
            self.Crawler.Href_link_list = []

            print(i, ':', self.page_urls_p)

            original_url_ = self.page_urls_p[i].split('/')[2]
            original_url = self.page_urls_p[i].split('/')[0] + '/'
            for a in range(1, len(self.page_urls_p[i].split('/')) - 1):
                original_url += self.page_urls_p[i].split('/')[a] + '/'

            pref_tag = self.pref_tags[i]
            pref_id = self.pref_IDs[i]
            text = self.Crawler.get_article_text_with_tag(url=self.page_urls_p[i], tag1='div', tag2=pref_tag,
                                                          tag3=pref_id)
            self.Crawler.get_a_list(url=self.page_urls_p[i], text=text)

            for j in range(len(self.Crawler.Href_link_list)):
                Link = str(self.Crawler.Href_link_list[j]).replace('amp;', '')

                if self.url_check(Link) is True:
                    Url = Link
                else:
                    if Link[0] == '/':
                        Url = original_url + Link
                    else:
                        Url = original_url + '/' + Link

                idx = self.search_page_c(original_url_)
                if idx != -1:
                    body_tag = self.body_tags[idx]
                    body_id = self.body_IDs[idx]

                    print(Url, body_tag, body_id)

                    body_text = self.Crawler.get_article_text(url=Url, tag1='div', tag2=body_tag,
                                                              tag3=body_id)
                    doc_file.write(str(self.Crawler.Href_title_list[j]).replace('\n', '') + '\n')
                    doc_file.write(body_text)
                    doc_file.write('\n' + '\a')

    def get_complete_url(self):
        idx = self.search_page_r(self.Url)
        if idx != -1:
            rule = str(self.rules[idx])

            for i in range(self.Rule_List.shape[0]):
                if self.Rule_Values[i] == '':
                    self.Complete_Url = 'http://' + self.Url
                    return 0
                rule = rule.replace(self.Rule_List[i], self.Rule_Values[i])

            self.Complete_Url = 'http://' + self.Url + rule

        else:
            self.Complete_Url = 'http://' + self.Url

    def process(self, command):
        TK = str(command).split()
        rule_idx = -1

        try:
            rule_idx = self.Command_Dic.index(TK[0])
        except:
            0
        print(rule_idx)
        if rule_idx != -1:
            code, result = self.Function_list[rule_idx](TK)

            return code, result

        return -1, -1

    def search_dictionary(self, word):
        try:
            idx = self.word_dic.index(word)
            return idx
        except:
            return -1

    def search_category(self, word):
        try:
            idx = self.page_names.index(word)
            return idx
        except:
            return -1

    def search_page(self, word):
        try:
            idx = self.page_names.index(word)
            return idx
        except:
            return -1

    def search_page_r(self, word):
        try:
            idx = self.page_urls_r.index(word)
            return idx
        except:
            return -1

    def search_page_c(self, word):
        try:
            idx = self.page_urls_c.index(word)
            return idx
        except:
            return -1

    def search_page_a(self, word):
        self.Crawler.Href_title_list = []
        self.Crawler.Href_link_list = []

        idxs = []
        index = 0

        while True:
            try:
                idx = self.page_urls_a.index(word, index)
                idxs.append(idx)
                index = idx + 1
                if index >= len(self.page_urls_a):
                    return idxs
            except:
                return idxs

    ########
    # Function의 자세한 기능은 개발자 문서에서 확인가능
    #######

    def Function_1(self, TK):
        #move Funtion
        page_name = TK[1]
        idx = self.search_page(page_name)
        print(idx)

        if idx != -1:
            self.Url = self.page_urls[idx]
            rule_idx = self.search_page_r(self.Url)
            if rule_idx != -1:
                self.Rule = self.rules[rule_idx]
                TK = str(self.rules_value[rule_idx]).split(',')

                self.Rule_List = np.array(TK, dtype='<U20')
                self.Rule_Values = np.zeros(shape=[len(TK)], dtype='<U20')

            self.Complete_Url = ''

            return 0, None
        else:
            return -1, None

    def Function_2(self, TK):
        rule_name = TK[1]
        word = TK[2]
        idx = self.search_dictionary(word)
        if idx != -1:
            word = self.word_values[idx]

        for i in range(self.Rule_List.shape[0]):
            if rule_name == self.Rule_List[i]:
                self.Rule_Values[i] = word

        return 0, None

    def Function_3(self, TK):
        #move Funtion
        for i in range(len(self.Rule_List.shape[0])):
            if self.Rule_List[i] == '[넘버]':
                try:
                    page_num = int(self.Rule_Values[i]) + 1
                    self.Rule_Values[i] = str(page_num)
                except:
                    return -1, None

    def Function_4(self, TK):
        search_text = ''
        for i in range(1, len(TK)):
            search_text += TK + ' '

        self.conn.send("".join(search_text).encode(encoding='utf-8'))  # 받은 데이터를 그대로 클라이언트에 전송
        data = self.conn.recv(1024)

        if not data:
            return -1, None
        else:
            qa_processed = str(data, encoding='utf-8').replace('\n', '').replace('  ', ' ').strip()
            print(qa_processed)

            doc_idx = self.agent.search(qa_processed.split(' '))
            doc = self.agent.get_doc(doc_idx)
            doc_processed = preprocess_document(doc)

            start_index, stop_index = self.classifier.Propagate_QA(Model=1, TK1=doc_processed.split(), TK2=qa_processed.split())

            log = ''
            tk = doc.split()
            for i in range(len(tk)):
                if i == start_index:
                    log += '('
                log += tk[i]

                if i == stop_index:
                    log += ')'
                log += ' '

            idx = self.agent.search(TK[1:len(TK) - 1])

            #리턴값 설정해야함


    def Function_5(self, TK):

        idxs = self.search_page_a(self.Url)
        if len(idxs) != 0:
            for idx in idxs:
                href_tag = self.href_tags[idx]
                href_id = self.href_IDs[idx]

                self.get_complete_url()
                print(self.Complete_Url)
                text = self.Crawler.get_article_text_with_tag(url=self.Complete_Url, tag1='div', tag2=href_tag, tag3=href_id)

                self.Crawler.get_a_list(url=self.Complete_Url, text=text)
                self.Href_Idx = 0
                print(self.Crawler.Href_title_list)
                if self.recommend_direction is True:
                    self.crawl_with_href()

            return 0, None
        else:
            return -1, None

    def Function_6(self, TK):
        # move Funtion
        if self.Href_Idx < len(self.Crawler.Href_title_list):
            idx = self.Href_Idx
            if self.recommend_direction is True:
                idx = self.arg_index[self.Href_Idx]
            self.Href_Idx += 1
            return 1, self.number_dic[idx] + '번' + self.Crawler.Href_title_list[idx]
        else:
            return 1, '목록이 더 이상 존재하지 않습니다'

    def Function_7(self, TK):
        # move Funtion
        print('this doc')

        if self.Href_Idx - 1 < len(self.Crawler.Href_title_list):
            Link = str(self.Crawler.Href_link_list[self.Href_Idx - 1]).replace('amp;', '')

            if self.recommend_direction is True:
                Link = str(self.Crawler.Href_link_list[self.arg_index[self.Href_Idx - 1]])

            if self.url_check(Link) is True:
                Url = Link
            else:
                if Link[0] == '/':
                    Url = 'http://' + self.Url + Link
                else:
                    Url = 'http://' + self.Url + '/' + Link
            self.Complete_Url = Url
            print(self.Complete_Url)
            return 0, None
        else:
            return 1, '목록이 더 이상 존재하지 않습니다'

    def Function_8(self, TK):
        # move Funtion
        try:
            idx = int(TK[1])
        except:
            return -1, None

        if idx < len(self.Crawler.Href_title_list):
            Link = str(self.Crawler.Href_link_list[idx])
            idx = Link.find('.')
            if idx != -1:
                self.Complete_Url = Link
            else:
                if Link[0] == '/':
                    self.Complete_Url = 'http://' + self.Url + '/' + Link
                else:
                    self.Complete_Url = 'http://' + self.Url + Link

            return 0, None
        else:
            return 1, '해당 번호의 문서가 존재하지 않습니다'

    def Function_9(self, TK):
        # move Funtion
        idx = self.search_page_c(self.Url)
        if idx != -1:
            body_tag = self.body_tags[idx]
            body_id = self.body_IDs[idx]
            if self.Complete_Url == '':
                self.get_complete_url()
            print(self.Complete_Url)
            body_text = self.Crawler.get_article_text(url=self.Complete_Url, tag1='div', tag2=body_tag, tag3=body_id)
            self.current_text = body_text
            return 1, body_text
        else:
            return -1, None

    def Function_10(self):
        recommend_file = open('recommend.txt', 'a', encoding='utf-8')
        recommend_file.write(self.current_text)

    def Function_11(self):
        non_recommend_file = open('non_recommend.txt', 'r', encoding='utf-8')
        non_recommend_file.write(self.current_text)

    def Function_12(self, TK):
        search_text = ''
        for i in range(1, len(TK)):
            search_text += TK + ' '

        self.conn.send("".join(search_text).encode(encoding='utf-8'))  # 받은 데이터를 그대로 클라이언트에 전송
        data = self.conn.recv(1024)

        if not data:
            return -1, None
        else:
            qa_processed = str(data, encoding='utf-8').replace('\n', '').replace('  ', ' ').strip()
            print(qa_processed)

            doc_idx = agent.search(qa_processed.split(' '))
            doc = agent.get_doc(doc_idx)

            return 1, doc

    def Function_13(self, TK):
        # move Funtion
        idxs = self.search_page_a(self.Url)
        if len(idxs) != 0:
            for idx in idxs:
                href_tag = self.href_tags[idx]
                href_id = self.href_IDs[idx]
                text = self.Crawler.get_article_text_with_tag(url=self.Complete_Url, tag1='div', tag2=href_tag,
                                                              tag3=href_id)
                self.Crawler.get_a_list(url=self.Complete_Url, text=text)
                self.Href_Idx = 0

                if self.recommend_direction is True:
                    self.crawl_with_href()

            return 0, None
        else:
            return -1, None

    def Function_14(self):
        if self.recommend_direction is True:
            self.recommend_direction = False
            return 1, '기본 순서로 설정되었습니다'
        else:
            self.recommend_direction = True
            return 1, '추천 순서로 설정되었습니다'

    def Function_15(self, TK):
        rule_name = '[' + TK[1] + ']'
        word = TK[2]
        idx = self.search_dictionary(word)
        if idx != -1:
            word = self.word_values[idx]

        for i in range(len(self.Rule_List.shape[0])):
            if rule_name == self.Rule_List[i]:
                self.Rule_Values[i] = word
