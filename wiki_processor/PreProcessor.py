import nltk
import WIKI_QA_DATASET
non_list = ['VBZ', 'EX', 'PRP', 'DT']

"""
result = ''

TK = 'As	at	most	other	universities	,	Notre	Dame	s	students	run	a	number	of	news	media	outlets		The	nine	student	-	run	outlets	include	three	newspapers	,	both	a	radio	and	television	station	,	and	several	magazines	and	journals		Begun	as	a	one	-	page	journal	in	September	1876	,	the	Scholastic	magazine	is	issued	twice	monthly	and	claims	to	be	the	oldest	continuous	collegiate	publication	in	the	United	States		The	other	magazine	,	The	Juggler	,	is	released	twice	a	year	and	focuses	on	student	literature	and	artwork		The	Dome	yearbook	is	published	annually		The	newspapers	have	varying	publication	interests	,	with	The	Observer	published	daily	and	mainly	reporting	university	and	other	news	,	and	staffed	by	students	from	both	Notre	Dame	and	Saint	Mary	s	College		Unlike	Scholastic	and	The	Dome	,	The	Observer	is	an	independent	publication	and	does	not	have	a	faculty	advisor	or	any	editorial	oversight	from	the	University		In	1987	,	when	some	students	believed	that	The	Observer	began	to	show	a	conservative	bias'.split('\t')
tagged_TK = nltk.pos_tag(TK)
for j in range(len(TK)):
    have_to_add = False
    try:
        print(tagged_TK[j][1][0], tagged_TK[j][1][1])
    except:
        0
    if tagged_TK[j][1][0] == 'V' and tagged_TK[j][1][1] == 'B':
        have_to_add = True
    if tagged_TK[j][1][0] == 'N' and tagged_TK[j][1][1] == 'N':
        have_to_add = True
    if tagged_TK[j][1][0] == 'J' and tagged_TK[j][1][1] == 'J':
        have_to_add = True
    for k in non_list:
        if tagged_TK[j][1] == k:
            have_to_add = False

    if have_to_add is True:
        result += tagged_TK[j][0]
        if j != len(TK) - 1:
            result += ' '
print(TK)
print()
print()
print()
print(result)
print('!!')
input()
"""

#refined_data
file_P = open('korean-english-park.train.en', 'r', encoding='utf8')
file_Q = open('refined_data_question', 'r', encoding='utf8')

new_file_P = open('new_korean-english-park.train.en', 'w', encoding='utf8')
new_file_Q = open('new_refined_data_question', 'w', encoding='utf8')

Paragraph = file_P.read().lower().split('\n')
Question = file_Q.read().lower().split('\a')

#paragraph
for i in range(len(Paragraph)):
    print(i)

    TK = Paragraph[i].split(' ')
    for a in range(len(TK)):
        if TK[a] == '':
            TK[a] = ' '
    if len(TK) > 1:
        #tagged_TK = nltk.pos_tag(TK)
        try:
            tagged_TK = nltk.pos_tag(TK)
        except:
            print(TK)
            input()

        for j in range(len(TK)):
            have_to_add = False
            if tagged_TK[j][1][0] == 'V' and tagged_TK[j][1][1] == 'B':
                have_to_add = True
            if tagged_TK[j][1][0] == 'N' and tagged_TK[j][1][1] == 'N':
                have_to_add = True
            if tagged_TK[j][1][0] == 'J' and tagged_TK[j][1][1] == 'J':
                have_to_add = True
            for k in non_list:
                if tagged_TK[j][1] == k:
                    have_to_add = False

            if have_to_add is True:
                new_file_P.write(tagged_TK[j][0])
                if j != len(TK) - 1:
                    new_file_P.write(' ')
    new_file_P.write('\n')
new_file_P.close()

input()
"""
#question
for i in range(len(Question)):
    print(i)

    TK = Question[i].split('\t')
    for a in range(len(TK)):
        if TK[a] == '':
            TK[a] = ' '

    if len(TK) > 1:
        tagged_TK = nltk.pos_tag(TK)
        for j in range(len(TK)):
            have_to_add = False
            if tagged_TK[j][1][0] == 'V' and tagged_TK[j][1][1] == 'B':
                have_to_add = True
            if tagged_TK[j][1][0] == 'N' and tagged_TK[j][1][1] == 'N':
                have_to_add = True
            if tagged_TK[j][1][0] == 'J' and tagged_TK[j][1][1] == 'J':
                have_to_add = True
            for k in non_list:
                if tagged_TK[j][1] == k:
                    have_to_add = False

            if have_to_add is True:
                new_file_Q.write(tagged_TK[j][0])
                if j != len(TK) - 1:
                    new_file_Q.write('\t')
    new_file_Q.write('\a')
new_file_Q.close()
"""
#wiki qa data
dataset = WIKI_QA_DATASET.dataprocessor()

new_file_P = open('WIKIQA_paragraph', 'w', encoding='utf8')
new_file_Q = open('WIKIQA_data_question', 'w', encoding='utf8')
new_file_L = open('WIKIQA_data_label', 'w', encoding='utf8')

Paragraph = file_P.read().lower().split('\a')
Question = file_Q.read().lower().split('\a')

#paragraph
for i in range(len(dataset.Questions)):
    TK = dataset.Paragraphs[i]
    tagged_TK = nltk.pos_tag(TK)
    for j in range(len(TK)):
        have_to_add = False
        if tagged_TK[j][1][0] == 'V' and tagged_TK[j][1][1] == 'B':
            have_to_add = True
        if tagged_TK[j][1][0] == 'N' and tagged_TK[j][1][1] == 'N':
            have_to_add = True
        if tagged_TK[j][1][0] == 'J' and tagged_TK[j][1][1] == 'J':
            have_to_add = True
        for k in non_list:
            if tagged_TK[j][1] == k:
                have_to_add = False

        if have_to_add is True:
            new_file_P.write(tagged_TK[j][0])
            if j != len(TK) - 1:
                new_file_P.write('\t')
    new_file_P.write('\a')
    new_file_L.write(str(dataset.Labels[i]))
    new_file_L.write('\a')

new_file_P.close()

#question
for i in range(len(dataset.Questions)):
    TK = dataset.Questions[i]
    tagged_TK = nltk.pos_tag(TK)
    for j in range(len(TK)):
        have_to_add = False
        if tagged_TK[j][1][0] == 'V' and tagged_TK[j][1][1] == 'B':
            have_to_add = True
        if tagged_TK[j][1][0] == 'N' and tagged_TK[j][1][1] == 'N':
            have_to_add = True
        if tagged_TK[j][1][0] == 'J' and tagged_TK[j][1][1] == 'J':
            have_to_add = True
        for k in non_list:
            if tagged_TK[j][1] == k:
                have_to_add = False

        if have_to_add is True:
            new_file_Q.write(tagged_TK[j][0])
            if j != len(TK) - 1:
                new_file_Q.write('\t')
    new_file_Q.write('\a')
new_file_Q.close()

a = 'i really fucking go home now'
a = a.split()
print(nltk.pos_tag(a)[0][0])

new_file_L.close()
new_file_P.close()
new_file_Q.close()
