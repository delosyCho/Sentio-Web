import numpy as np
import codecs
import pandas

from functions import *

exo_paragraph = open('exo_paragraph__', 'w', encoding='utf-8')
exo_question = open('exo_question_', 'w', encoding='utf-8')
exo_label = open('exo_label_', 'w', encoding='utf-8')

exo_paragraph_ = open('exo_paragraph_', 'r', encoding='utf-8')
exo_question_ = open('exo_question', 'r', encoding='utf-8')
exo_label_ = open('exo_label', 'r', encoding='utf-8')
exo_answers = open('exo_answers', 'r', encoding='utf-8')
# 정보가 들어있는 텍스트파일

answers = exo_answers.read().split('\n')

exo_Questions = exo_question_.read().split('\a')
exo_answer_info = exo_paragraph_.read()#.split('@#!')
exo_answer = exo_label_.read().split('\a')

for i in range(len(answers)):
    line = '# ' + str(i) + ' @'
    #print(line)
    exo_answer_info = exo_answer_info.replace(line, answers[i])

exo_answer_info = exo_answer_info.split('@#!')
#print(exo_answer_info[0])
#input()

count = 0

for i, paragraph in enumerate(exo_answer_info):
    temp_TK = str(exo_answer_info[i]).replace('\n', ' ').replace('   ', ' ').replace('  ', ' ').replace('  ', ' ').split()

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
                TK = temp_TK
                print(TK[start_index], ':', TK[stop_index], ':', exo_answer[i])
                print(exo_answer_info[i])
                for k in range(len(TK)):
                    exo_paragraph.write(TK[k].strip() + ' ')
                exo_paragraph.write('@#!\n')

                exo_question.write(exo_Questions[i])
                exo_question.write('\n')

                exo_label.write(str(start_index) + '#' + str(stop_index))
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