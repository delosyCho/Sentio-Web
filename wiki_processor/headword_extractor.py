# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
import numpy as np

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

type_check_Tokens = ['틀:', '분류:', '위키백과:']

check_Tokens = ['| 출생일', '| 출생지',
                '|출생일', '|출생지']

check_Tokens2 = ['|면적', '|세대', '|총인구', '|인구', '|행정구역',
                 '| 면적', '| 세대', '| 총인구', '| 인구', '| 행정구역',
                 '|국기', '|수도', '|통화', '|국제 전화', '|도메인', '|시간대',
                 '| 국기', '| 수도', '| 통화', '| 국제 전화', '| 도메인', '| 시간대']

check_Tokens3 = '|계=[[식물|식물계]]'

check_Tokens4 = '|계=[[동물|동물계]]'

check_Tokens5 = ['|학부', '|대학원', '|교직원', '|총장',
                 '| 학부', '| 대학원', '| 교직원', '| 총장']

check_Tokens6 = []


""" Test가 필요할 때 """
if True:
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
                            correct_coefficient = 0

                            try:
                                TK = text.text.split('\n')
                            except:
                                print(text.text)
                                # input()

                            if title == '맥스웰 방정식':
                                while True:
                                    print('!!')
                                    print(text.text)
                                    input()
                                    ani_pla_coefficient = -1
                                    lines = text.text.split('\n')
                                    print(len(check_Tokens3))
                                    for l in lines:
                                        for j in range(len(check_Tokens2)):
                                            if len(l) > len(check_Tokens2[j]):
                                                if l[0:len(check_Tokens2[j])] == check_Tokens2[j]:
                                                    print('captured', l)
                                                    ani_pla_coefficient += 1
                                                    #input()

                                    print('testresult', ani_pla_coefficient)
                                    input()

                            if title == '#':
                                while True:
                                    print('!!')
                                    #print(text.text)
                                    TK = text.text.split('\n')
                                    input()

name_entity_file = open('hw_names', 'w', encoding='utf-8')
place_entity_file = open('hw_places', 'w', encoding='utf-8')
animal_entity_file = open('hw_animals', 'w', encoding='utf-8')
plants_entity_file = open('hw_plants', 'w', encoding='utf-8')
univ_entity_file = open('hw_univ', 'w', encoding='utf-8')

non_name_entity_file = open('non', 'w', encoding='utf-8')

# 11 split file
for a in range(11):
    print(a, 'processing...')

    tree = ET.parse('wikisplit' + str(a))
    root = tree.getroot()

    for page in root:
        title = ''

        check_ok = True

        for tags in page:
            if tags.tag == 'title':
                title = tags.text

                for j in range(1, len(type_check_Tokens)):
                    if len(title) > len(type_check_Tokens[j]):
                        if title[0:len(type_check_Tokens[j])] == type_check_Tokens[j]:
                            check_ok = False

                if len(title) > len(type_check_Tokens[0]):
                    if title[0:len(type_check_Tokens[0])] == type_check_Tokens[0]:
                        title = title[2:len(title) - 2]

            if tags.tag == 'revision' and check_ok is True:
                for text in tags:
                    if text.tag == 'text':
                        correct_coefficient = 0
                        correct_coefficient2 = 0
                        ani_pla_coefficient = -1
                        correct_coefficient3 = 0
                        correct_coefficient4 = 0

                        TK = 0
                        try:
                            TK = text.text.split('\n')
                        except:
                            print(text.text)
                            #input()

                        check_array1 = np.zeros(shape=[len(check_Tokens)], dtype='i')
                        check_array2 = np.zeros(shape=[len(check_Tokens2)], dtype='i')
                        check_array3 = np.zeros(shape=[len(check_Tokens5)], dtype='i')
                        check_array4 = np.zeros(shape=[len(check_Tokens6)], dtype='i')

                        if TK != 0:
                            for i in TK:
                                i = i.strip()

                                if title == '#':
                                    print('check', i)
                                    input()

                                for j in range(len(check_Tokens)):
                                    if len(i) > len(check_Tokens[j]):
                                        if i[0:len(check_Tokens[j])] == check_Tokens[j]:
                                            check_array1[j] = 1

                                for j in range(len(check_Tokens2)):
                                    if len(i) > len(check_Tokens2[j]):
                                        if i[0:len(check_Tokens2[j])] == check_Tokens2[j]:
                                            check_array2[j] = 1

                                for j in range(len(check_Tokens5)):
                                    if len(i) > len(check_Tokens5[j]):
                                        if i[0:len(check_Tokens5[j])] == check_Tokens5[j]:
                                            check_array3[j] = 1

                                for j in range(len(check_Tokens6)):
                                    if len(i) > len(check_Tokens6[j]):
                                        if i[0:len(check_Tokens6[j])] == check_Tokens6[j]:
                                            if title == '#':
                                                print('check', check_Tokens6[j])
                                                if j > len(check_Tokens6) / 2:
                                                    print('!!', j - int(len(check_Tokens6) / 2))
                                                else:
                                                    print('@@', j)
                                                input()
                                            check_array4[j] = 1

                                if len(i) >= len(check_Tokens3):
                                    if i[0:len(check_Tokens3)] == check_Tokens3:
                                        ani_pla_coefficient = 0

                                if len(i) >= len(check_Tokens4):
                                    if i[0:len(check_Tokens4)] == check_Tokens4:
                                        ani_pla_coefficient = 1

                        if title == '조j제 모리뉴':
                            print('Check!!!!!!!!')
                            print(check_array1.sum())
                            print(check_array2.sum())
                            print(check_array3.sum())
                            print(check_array4.sum())
                            print(text.text)
                            input()

                        if ani_pla_coefficient == 0:
                            #print('y:', title)
                            plants_entity_file.write(title)
                            plants_entity_file.write('\n')
                        elif ani_pla_coefficient == 1:
                            #print('y:', title)
                            animal_entity_file.write(title)
                            animal_entity_file.write('\n')
                        elif check_array1.sum() > 1:
                            #print('y:', title)
                            name_entity_file.write(title)
                            name_entity_file.write('\n')
                        elif check_array2.sum() > 1:
                            #print('y:', title)
                            place_entity_file.write(title)
                            place_entity_file.write('\n')
                        elif check_array3.sum() > 1:
                            #print('y:', title)
                            univ_entity_file.write(title)
                            univ_entity_file.write('\n')
                        elif check_array4.sum() > 1:
                            #print('y:', title)
                            place_entity_file.write(title)
                            place_entity_file.write('\n')
                        else:
                            #print('n:', title)
                            non_name_entity_file.write(title)
                            non_name_entity_file.write('\n')

name_entity_file.close()
non_name_entity_file.close()
