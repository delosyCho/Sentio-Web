# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

""" Test가 필요할 때 """
if False:
    for i in range(11):
        print(i)

        tree = ET.parse('wikisplit' + str(i))
        root = tree.getroot()

        for page in root:
            title = ''

            for tags in page:
                if tags.tag == 'title':
                    title = tags.text

                if tags.tag == 'revision':
                    for text in tags:
                        if text.tag == 'text':

                            if title == '덩샤오핑':
                                print(text.text)
                                input()


def Bracket_Processing(string, start, end):
    result = ''

    is_bracket = False

    for i in range(len(string)):
        if string[i] == start:
            is_bracket = True
        elif string[i] == end:
            is_bracket = False
        elif is_bracket == False:
            result += string[i]

    return result


def Bracket_Processing2(string, start, end):
    result = ''

    is_bracket = False

    for i in range(len(string)):
        if string[i] == start:
            is_bracket = True
        elif string[i] == end:
            is_bracket = False
            result += string[i]
        elif is_bracket == False:
            result += string[i]

    return result


def Bracket_Processing3(string, start, end):
    result = ''

    is_bracket = False

    for i in range(len(string)):
        if i + 1 < len(string):
            if string[i] == start[0] and string[i + 1] == start[1]:
                is_bracket = True
            elif string[i] == end:
                is_bracket = False
                result += string[i]
            elif is_bracket == False:
                result += string[i]
        else:
            if string[i] == end:
                is_bracket = False
                result += string[i]
            elif is_bracket == False:
                result += string[i]
    return result


def Bracket_Processing4(string, start, end):
    result = ''

    bracket_count = 0
    is_bracket = False

    #print(len(string))

    for i in range(len(string)):
        if i + 1 < len(string) and i > 0:
            if string[i] == start[0] and string[i + 1] == start[1]:
                bracket_count += 1
            elif string[i - 1] == end[0] and string[i] == end[1]:
                bracket_count -= 1
            elif bracket_count == 0:
                result += string[i]
        elif i == 0:
            if string[i] == start[0] and string[i + 1] == start[1]:
                bracket_count += 1
            else:
                result += string[i]
        else:
            if bracket_count == 0:
                result += string[i]
        #print(string[i])
        #print(bracket_count)

    return result
"""
인명 인식을 위한 토큰 요소

2. | 원래 이름
3. | 출생일
4. | 출생지
5. | 국적
6. | 직업
7. | 소속
8. | 배우자
9. | 종교
10. | 학력
11. | 부모
"""

"""
지명 인식을 위한 토큰 요소

1. |인구
2. |지도
3. |면적
4. |세대
5. |총인구
6. |인구밀도
7. |행정구역

"""

"""
동/식물명 인식을 위한 토큰 요소

1. |계: 식물계
2. |계: 동물계

"""


def check_hangeul(z):
    a = '가'
    b = '힣'
    c = 'A'
    d = 'z'
    e = '1'
    f = '0'

    if a <= z <= b:
        #print('!!')
        #print(c)
        #input()
        return True
    elif e <= z <= f:
        return True
    else:
        #print('@@')
        #print(c)
        #input()
        return False


def check(line):
    for i in range(len(line)):
        #print(line[i])

        if line[i] == ':' and len(line) > 2:
            if line[i - 2] == '파' and line[i - 1] == '일':
                return False

        if line[i] == ':' and len(line) > 1:
            if line[i - 1] == '틀':
                return False

        if line[i] == ':' and len(line) > 4:
            if line[i - 4] == '위' and line[i - 3] == '키' and line[i - 2] == '백' and line[i - 1] == '과':
                return False

        if line[i] == ':' and len(line) > 2:
            if line[i - 2] == '분' and line[i - 1] == '류':
                return False

    return True

def preprocess_text(sen):
    result_str = ''



    for i in range(len(sen)):
        if len(sen[i]) > 1:
            if sen[i][0] == '=':
                0
            elif sen[i][0] == '{':
                0
            elif sen[i][0] == '}':
                0
            elif sen[i][0] == '|':
                0
            elif sen[i][0] == ' ':
                0
            elif sen[i][0] == '[':
                0
            elif sen[i][0] == '*':
                0
            elif sen[i][0] == '<':
                0
            elif sen[i][0] == ':':
                0
            elif sen[i][0] == '#':
                0
            elif sen[i][0] == '!':
                0
            elif sen[i][0] == ';':
                0
            else:
                if len(sen[i]) > 3:
                    if sen[i][0] == '파' and sen[i][1] == '일' and sen[i][2] == ':':
                        0
                    elif sen[i][0] == 's' and sen[i][1] == 't' and sen[i][2] == 'y':
                        0
                    else:
                        result_str += sen[i] + '\n'

    return result_str


def preprocess_text_(result_str):
    """

    :param result_str:
    :return:
    """
    result_str = Bracket_Processing4(result_str, '{{', '}}')
    result_str = Bracket_Processing(result_str, '「', '」')
    result_str = Bracket_Processing3(result_str, ' (', ')')
    result_str = Bracket_Processing(result_str, '(', ')')
    result_str = Bracket_Processing(result_str, '<', '>')
    result_str = Bracket_Processing(result_str, '[', ']')
    result_str = Bracket_Processing2(result_str, '|', ' ')
    ###########

    result_str = result_str.replace('-', ' ')
    result_str = result_str.replace('\'\'\'', '')
    result_str = result_str.replace('”', '')
    result_str = result_str.replace('“', '')
    result_str = result_str.replace('《', '')
    result_str = result_str.replace('》', '')
    result_str = result_str.replace(' ( ', '')
    result_str = result_str.replace(' ) ', '')
    result_str = result_str.replace(' \" ', '')
    result_str = result_str.replace('\"', '')
    result_str = result_str.replace('.', '')
    result_str = result_str.replace('[[', '')
    result_str = result_str.replace(']]', '')
    result_str = result_str.replace(',', '')
    result_str = result_str.replace('\'\'', '')
    result_str = result_str.replace('|', ' ')

    result_str = Bracket_Processing4(result_str, '{{', '}}')
    result_str = Bracket_Processing(result_str, '「', '」')
    # result_str = Bracket_Processing3(result_str, ' (', ')')
    result_str = Bracket_Processing(result_str, '(', ')')
    result_str = Bracket_Processing(result_str, '<', '>')
    result_str = Bracket_Processing(result_str, '[', ']')

    return result_str

data = pd.read_excel("C:\\Users\\Administrator\\Desktop\\qadataset\\exobrain\\"\
                    "엑소브레인 말뭉치V3.0\\엑소브레인 QA Datasets\\ETRI QA Datasets\\"\
                    "단문질문 QA dataset(위키피디아, 기본태깅, 1,776개).xlsx")

title_list = []
answer_list = []
answer_ground_list = []
question_list = []

for i in range(len(data['위키피디아 제목']) - 1):
    title_list.append(data['위키피디아 제목'][i])

    answer = preprocess_text_(str(data['정답 근거1(문장)'][i]))
    answer_ground_list.append(answer)

    answer_list.append(str(data['정답'][i]))

    que = str(data['질문'][i]).replace('?', '')
    question_list.append(que)

title = ''
refined_text = ''
refined_text_list = []

Wiki_Index = -1
Wiki_answer = 0
Wiki_question = 0
Wiki_answer_ground = 0

# 11개 split file
for a in range(11):
    print(a, 'processing...')

    tree = ET.parse('wikisplit' + str(a))
    root = tree.getroot()

    for page in root:
        check_ok = True

        for tags in page:
            if tags.tag == 'title':
                title = tags.text

                if Wiki_Index != -1:
                    print(title)
                    print('-------')
                    print(refined_text)
                    print(refined_text.find(Wiki_answer_ground), Wiki_answer_ground)
                    input()
                    refined_text = ''

                Wiki_Index = -1
                for t in range(len(title_list)):
                    if title_list[t] == title:
                        Wiki_Index = t
                        Wiki_answer = answer_list[t]
                        Wiki_question = question_list[t]
                        Wiki_answer_ground = answer_ground_list[t]

                        print('index::', title_list[t], title)

            if Wiki_Index != -1:
                if tags.tag == 'revision' and check_ok is True:
                    for text in tags:
                        if text.tag == 'text':
                            #print(text.text)

                            result_str = ''

                            text_ = str(text.text)
                            #text_ = preprocess_text_(text_)

                            sen = text_.split('\n')
                            for i in range(len(sen)):
                                if len(sen[i]) > 1:
                                    if sen[i][0] == '=' and sen[i][1] != '=':
                                        0
                                    elif sen[i][0] == '{' and sen[i][1] != '{':
                                        0
                                    elif sen[i][0] == '}' and sen[i][1] != '}':
                                        0
                                    elif sen[i][0] == '|' and sen[i][1] != '|':
                                        0
                                    elif sen[i][0] == ' ' and sen[i][1] != ' ':
                                        0
                                    elif sen[i][0] == '[' and sen[i][1] != '[':
                                        0
                                    elif sen[i][0] == '*' and sen[i][1] != '*':
                                        0
                                    elif sen[i][0] == '<' and sen[i][1] != '<':
                                        0
                                    elif sen[i][0] == ':' and sen[i][1] != ':':
                                        0
                                    elif sen[i][0] == '#' and sen[i][1] != '#':
                                        0
                                    elif sen[i][0] == '!' and sen[i][1] != '!':
                                        0
                                    elif sen[i][0] == ';' and sen[i][1] != ';':
                                        0
                                    else:
                                        if len(sen[i]) > 3:
                                            if sen[i][0] == '파' and sen[i][1] == '일' and sen[i][2] == ':':
                                                0
                                            elif sen[i][0] == 's' and sen[i][1] == 't' and sen[i][2] == 'y':
                                                0
                                            else:
                                                result_str += sen[i] + '\n'

                            if title == '덩샤오핑':
                                for k in range(len(sen)):
                                    print(sen[k])
                                    input()
                            result_str = preprocess_text_(result_str)

                            #print(result_str)
                            #input()

                            TK = result_str.split('\n')
                            for k in range(len(TK)):
                                if TK[k] != '':
                                    if len(TK[k]) > 0:
                                        if check_hangeul(TK[k][0]) is True:
                                            refined_text += ' ' + TK[k]
                                            #print("!!")
                                            #print(TK[k])
                                            #input()
                                        else:
                                            0
                                            #print("!!")
                                            #print(result_str)
                                            #input()
