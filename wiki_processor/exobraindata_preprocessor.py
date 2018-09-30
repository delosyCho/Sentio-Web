import numpy as np
import codecs
import pandas

from functions import *

exo_paragraph = open('exo_paragraph', 'w', encoding='utf-8')
exo_question = open('exo_question', 'w', encoding='utf-8')
exo_label = open('exo_label', 'w', encoding='utf-8')
exo_answers = open('exo_answers', 'w', encoding='utf-8')

paragraph_file = open('wiki_corpus', 'r', encoding='utf-8')
rule_file = open('wiki_info', 'r', encoding='utf-8')
# 정보가 들어있는 텍스트파일

exo_Questions = []
exo_Titles = []
exo_answer_info = []
exo_answer = []

exobrain_data1 = pandas.read_excel('exo1.xlsx')
temp = exobrain_data1['질문']
temp2 = exobrain_data1['위키피디아 제목']
temp3 = exobrain_data1['정답 근거1(문장)']
temp4 = exobrain_data1['정답']

for i in range(len(temp)):
    exo_Questions.append(str(temp[i]).replace('?', ''))
    exo_Titles.append(str(temp2[i]))

    info_str = str(temp3[i])

    exo_answer_info.append(preprocess(info_str))
    exo_answer.append(preprocess(str(temp4[i])))

exobrain_data1 = pandas.read_excel('exo3.xlsx')
temp = exobrain_data1['질문']
temp2 = exobrain_data1['위키피디아 제목']
temp3 = exobrain_data1['정답 근거1(문장)']
temp4 = exobrain_data1['정답']

for i in range(len(temp)):
    exo_Questions.append(str(temp[i]).replace('?', ''))
    exo_Titles.append(str(temp2[i]))

    info_str = str(temp3[i]).replace('.', '\n')

    exo_answer_info.append(preprocess(info_str))
    exo_answer.append(preprocess(str(temp4[i])))

print(len(exo_answer))
print(len(exo_answer_info))
print(len(exo_Questions))
print(len(exo_Titles))

exo_data_dictionary = np.array(exo_Titles, dtype='<U20')
exo_dictionary_index = exo_data_dictionary.argsort()
exo_data_dictionary.sort()

paragraphs = paragraph_file.read().split('\a')

count = 0

for i in range(len(exo_answer)):
    exo_answers.write(exo_answer[i])
    exo_answers.write('\n')
exo_answers.close()

for i, paragraph in enumerate(exo_answer_info):
    temp_TK = str(exo_answer_info[i]).split()

    TK = str(exo_answer[i]).split()

    if len(TK) > 0:
        start_word = TK[0]
        stop_word = TK[len(TK) - 1]

        TK = str(exo_answer_info[i]).split()
        start_index = -1
        stop_index = -1

        for j in range(len(TK)):
            a = TK[j].find(start_word)
            b = TK[j].find(stop_word)

            if a != -1 and start_index == -1:
                start_index = j
            if b != -1 and stop_index == -1:
                stop_index = j

        if start_index != -1 and stop_word != -1:
            if start_index <= stop_index:
                para = str(exo_answer_info[i]).replace(str(exo_answer[i]), '#' + str(i) + '@')
                TK = para.split('\n')
                for k in range(len(TK)):
                    exo_paragraph.write(TK[k].strip())
                    exo_paragraph.write('\n')
                exo_paragraph.write('@#!\n')

                exo_question.write(exo_Questions[i])
                exo_question.write('\a')

                exo_label.write(exo_answer[i])
                exo_label.write('\a')

                count += 1
            else:
                print('check!!!!!!!!!!')
                print(exo_answer_info[i])
                print(exo_answer[i])
                print(start_index, stop_index)
                print('---------------')

print(count)

exo_paragraph.close()
exo_question.close()
exo_label.close()