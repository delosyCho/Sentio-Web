filename = 'gen_paragraph__p__pre__r_____model'
filename_ = 'gen_label_model'
filename__ = 'gen_paragraph__p__pre__r____'
filename___ = 'gen_question'


file = open(filename, 'r', encoding='utf-8')
file_ = open(filename_, 'r', encoding='utf-8')
file__ = open(filename__, 'r', encoding='utf-8')
file___ = open(filename___, 'r', encoding='utf-8')

result = open(filename + '_temp', 'w', encoding='utf-8')
result_ = open(filename_ + '_temp', 'w', encoding='utf-8')
result__ = open(filename___ + '_temp', 'w', encoding='utf-8')

paragraphs = file.read().replace(' @ ', ' @').replace(' # ', '# ').split('@#!')
paragraphs2 = file__.read().replace(' @ ', ' @').replace(' # ', '# ').split('@#!')

labels = file_.read().replace('@ ', '@').replace(' #', '#').split('\a')
questions = file___.read().split('\a')

print(len(paragraphs))
print(len(labels))
print(len(questions))

P_Length = 200

count = 0
wrong = 0
wrong2 = 0

for i in range(len(labels) - 1):
    sentences = paragraphs[i].split('\n')
    answer = labels[i].split(',')[0].strip()
    sentence_index = int(labels[i].split(',')[1].strip()) + 1

    start_word = answer.split()[0]
    stop_word = answer.split()[len(answer.split()) - 1]
    new_sentence = ''

    temp_paragraph = paragraphs[i].replace('\n', ' ')
    paragraph = paragraphs[i].replace('\n', ' ')
    paragraph2 = paragraphs2[i].replace('\n', ' ')

    start_index = temp_paragraph.find(start_word)
    stop_index = temp_paragraph.find(stop_word)

    list_str = list(paragraph)
    list_str[start_index] = '뷁'
    list_str[stop_index] = '붥'

    temp_paragraph = "".join(list_str)

    if start_index != -1 and stop_index != -1 and start_index <= stop_index:
        TK = temp_paragraph.split(' ')
        if int(stop_index) + 10 < P_Length:
            start_label = -1
            stop_label = -1

            for j in range(len(TK)):
                if len(TK[j]) > 0:
                    if TK[j][0] == '뷁':
                        start_label = j
                    if TK[j][0] == '붥':
                        stop_label = j

            if start_label == -1:
                start_label = stop_label

            if start_label != -1 and stop_label != -1:
                result.write(paragraph2 + '\a')
                result_.write(str(start_label) + '#' + str(stop_label) + '\a')
                result__.write(questions[i] + '\a')

                count += 1
        else:
            wrong += 1
    else:
        #print(start_index, stop_index)
        #print(start_word, stop_word)
        #print(sentences[sentence_index], sentence_index)
        #print(sentences)
        #input()
        wrong2 += 1

result.close()
result_.close()
result__.close()

print()
print(count)
print(wrong)
print(wrong2, '\n')

result = open(filename + '_temp', 'r', encoding='utf-8')
lines = result.read().split('\a')
print(len(lines))