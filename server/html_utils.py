import requests
from bs4 import BeautifulSoup
import urllib.request


class Web_Crawler:
    def __init__(self):
        self.ID_list = []
        self.Class_list = []
        self.Href_link_list = []
        self.Href_title_list = []

    def load_data(self):
        filename = ''

    def check_hangeul(self, z):
        a = '가'
        b = '힣'
        c = 'A'
        d = 'z'
        e = '1'
        f = '0'

        if a <= z <= b:
            # print('!!')
            # print(c)
            # input()
            return True
        elif c <= z <= d:
            return False
        elif e <= z <= f:
            return True
        else:
            # print('@@')
            # print(c)
            # input()
            return False

    def get_text(self, url, id, tag='id'):
        source_code_from_URL = urllib.request.urlopen(url)
        soup = BeautifulSoup(source_code_from_URL, 'lxml', from_encoding='utf-8')

        result = ''

        for item in soup.find_all('div', {tag: id}):
            result += str(item.find_all(text=True))

        return result

    def get_html(self, url):
       _html = ""
       resp = requests.get(url)
       if resp.status_code == 200:
          _html = resp.text
       return _html

    def get_a_list(self, url, text=None):
        current_index = 0

        if text is None:
            html = self.get_html(url)
        else:
            html = text
        while True:
            start_index = html.find("<a href=\"", current_index)
            start_index_ = html.find("\">", start_index)
            if start_index != -1:
                end_index = html.find("\"", start_index + len("<a href=\""))
                self.Href_link_list.append(html[start_index + len("<a href=\""):end_index])

                end_index_ = html.find("</a>", start_index_ + len("\">"))
                self.Href_title_list.append(html[start_index_ + len("\">"):end_index_])

                current_index = end_index + 1
            else:
                break

    def get_div_list(self, url):
        current_index = 0
        html = self.get_html(url)
        while True:
            start_index = html.find("<div id=\"", current_index)
            if start_index != -1:
                end_index = html.find("\"", start_index + 9)
                self.ID_list.append(html[start_index + 9:end_index])
                current_index = end_index + 1
            else:
                break

    def get_class_list(self, url):
        current_index = 0
        html = self.get_html(url)
        while True:
            start_index = html.find("<div class=\"", current_index)
            if start_index != -1:
                end_index = html.find("\"", start_index + len('<div class=\"'))
                try:
                    self.Class_list.index(html[start_index + len('<div class=\"'):end_index])
                except:
                    self.Class_list.append(html[start_index + len('<div class=\"'):end_index])
                current_index = end_index + 1
            else:
                break

    def get_score(self, url, string):
        str_TK = string.split()

        selected_ID = ''
        max_ = -99

        for id_ in self.ID_list:
            score = 0

            refined_text = []

            text = self.get_text(url, id_)

            text = text.replace('\'', '')
            text = text.replace(',', '')
            text = str(text)

            TK = text.split(' ')
            # print(text)
            for T in TK:
                if self.check_hangeul(T) is True:
                    # print('@@', T)
                    # input()
                    try:
                        searched_index = refined_text.index()
                    except:
                        searched_index = -1

                    if searched_index == -1:
                        refined_text.append(T)
                else:
                    0
                    # print('!!', T)
                    # input()

            for T in refined_text:
                for K in str_TK:
                    if T == K:
                        score += 1.0

            if len(refined_text) > 0:
                score /= len(text)

            #print(id_)
            #print(refined_text)
            #input()

            if score > max_:
                max_ = score
                selected_ID = id_

            print(id_, ':', score)

        return selected_ID, max_

    def get_score_by_class(self, url, string):
        str_TK = string.split()

        selected_ID = ''
        max_ = -99

        for id_ in self.Class_list:
            score = 0

            refined_text = []

            text = self.get_text(url, id_, tag='class')

            text = text.replace('\'', '')
            text = text.replace(',', '')
            text = str(text)

            TK = text.split(' ')
            # print(text)
            for T in TK:
                if self.check_hangeul(T) is True:
                    # print('@@', T)
                    # input()
                    try:
                        searched_index = refined_text.index()
                    except:
                        searched_index = -1

                    if searched_index == -1:
                        refined_text.append(T)
                else:
                    0
                    # print('!!', T)
                    # input()

            for T in refined_text:
                for K in str_TK:
                    if T == K:
                        score += 1.0

            if len(refined_text) > 0:
                score /= len(text)

            if score > max_:
                max_ = score
                selected_ID = id_

            print(id_, ':', score)

        return selected_ID, max_

    def get_most_contain_tag(self, url, text):
        self.get_div_list(url=url)
        selected_i, i_score = self.get_score(url=url, string=text)

        self.get_class_list(url=url)
        selected_c, c_score = self.get_score_by_class(url=url, string=text)

        if i_score > c_score:
            return 'id', selected_i
        else:
            return 'class', selected_c


    def get_article_text(self, url, tag1, tag2, tag3):
        source_code_from_URL = urllib.request.urlopen(url)
        soup = BeautifulSoup(source_code_from_URL, 'lxml', from_encoding='utf-8')
        text = str(soup.find_all(tag1, {tag2: tag3}))
        text = clean_script_tag(text, '<script', '</script>')
        text = clean_tag(text)
        text = replace_vowel(text)
        text = postprocess_text(text)
        return text

    def get_article_text_with_tag(self, url, tag1, tag2, tag3):
        source_code_from_URL = urllib.request.urlopen(url)
        soup = BeautifulSoup(source_code_from_URL, 'lxml', from_encoding='utf-8')
        text = str(soup.find_all(tag1, {tag2: tag3}))
        return text


def is_hanguel_sentence(string):
    utf_start_hanguel = 44032
    utf_end_hanguel = 55203

    utf_start_hanguel_ = 12593
    utf_end_hanguel_ = 12643

    h_cnt = 0
    n_cnt = 0

    for i in range(len(string)):
        if (utf_start_hanguel <= ord(string[i]) <= utf_end_hanguel) or (
                utf_start_hanguel_ <= ord(string[i]) <= utf_end_hanguel_):
            h_cnt += 1
        else:
            n_cnt += 1

    if h_cnt > n_cnt:
        return True
    else:
        return False


def postprocess_text(text):
    text = str(text)
    result = ''

    lines = text.split('\n')
    for i in range(len(lines)):
        if len(lines[i]) > 3 and is_hanguel_sentence(lines[i]):
            result += lines[i] + '\n'

    return result


def replace_vowel(text):
    text = str(text)
    dic = ['ㅋ', 'ㅎ', 'ㄱ', 'ㅉ', 'ㄴ', 'ㄷ', 'ㅜ', 'ㅠ']
    rep = ['크', '히', '고', '쯔', '노', '덜', '흑', '흑']

    for i in range(len(dic)):
        text = text.replace(dic[i], rep[i])

    return text


def clean_script_tag(text, tag_name, tag_name_):
    result = ''
    is_bracket = False
    for i in range(len(text)):
        try:
            if text[i:i + len(tag_name)] == tag_name:
                is_bracket = True
        except:
            0
        if is_bracket is False:
            result += text[i]
        try:
            if text[i - len(tag_name_):i] == tag_name_:
                is_bracket = False
        except:
            0
    return result


def clean_tag(text):
    result = ''
    is_bracket = False
    for i in range(len(text)):
        if text[i] == '<':
            is_bracket = True
        if is_bracket is False:
            result += text[i]
        if text[i] == '>':
            is_bracket = False
    return result

def get_word(text, index):
    result = ''
    is_bracket = False

    for i in range(index, len(text)):
        if is_bracket is True:
            if text[i] == '"':
                return result
            result += text[i]

        if text[i] == '"':
            is_bracket = True

    return False


def get_tag_by_index(text, index):
    start_index = -1
    stop_index = -1

    if index > 1:
        temp = index - 1

        for i in range(0, index):
            if text[temp] == '<':
                start_index = temp
                break
            else:
                temp -= 1

        for i in range(index + 1, len(text)):
            if text[i] == '>':
                stop_index = i
                break

    if start_index == -1 or stop_index == -1:
        return ''

    return text[start_index:stop_index + 1]

def get_text(URL):
    source_code_from_URL = urllib.request.urlopen(URL)
    soup = BeautifulSoup(source_code_from_URL, 'lxml', from_encoding='utf-8')
    text = ''
    for item in soup.find_all('div'):
        text = text + str(item.find_all(text=True))
        print('----------------------------')
        print(text)
        input()
    return text


def get_html(url):
    _html = ""
    resp = requests.get(url)
    if resp.status_code == 200:
        _html = resp.text
    return _html



def check_time(string):
    if len(string) < 2:
        return False

    dics = ['년', '월', '일', '시', '분', '초']

    is_Dic = False

    if len(string) > 2:
        for dic in dics:
            if string[len(string) - 2] == dic:
                is_Dic = True

    if is_Dic is True:
        for i in range(len(string) - 2):
            if not ('0' <= string[i] <= '9'):
                return False

    if is_Dic is True:
        return True

    for dic in dics:
        if string[len(string) - 1] == dic:
            is_Dic = True

    if is_Dic is False:
        return False

    if is_Dic is True:
        for i in range(len(string) - 1):
            if not ('0' <= string[i] <= '9'):
                return False

    return True


def check_number(string):
    if len(string) < 1:
        return False

    for i in range(len(string)):
        if not ('0' <= string[i] <= '9'):
            return False
    return True

def preprocess_document(string):
    result = ''

    lines = string.split('\n')
    for k in range(len(lines)):
        TK = lines[k].split(' ')

        for j in range(len(TK)):
            if check_time(TK[j]) is True:
                TK[j] = '###'
            elif check_number(TK[j]) is True:
                TK[j] = '#@#'

        result += TK[0]
        for j in range(1, len(TK)):
            line = ' ' + TK[j]
            result += line
        result += ' '
    return result