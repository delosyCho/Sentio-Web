import numpy as np

file = open('C:\\Users\\Administrator\\Desktop\\qadataset\\EXOBRAIN_NE_CORPUS_10000.txt', 'r', encoding='utf8')
result_file = open('NE_dataset', 'w', encoding='utf8')

strings = file.read()

word = ''
label = ''
start = False
end = False

for i in range(len(strings)):
    if strings[i] == '<':
        start = True
    elif strings[i] == '>':
        line = word + '' + label + '\n'
        result_file.write(line)

        start = False
        end = False
        word = ''
        label = ''
    elif start is True:
        word += strings[i]
    elif strings[i] == ':':
        start = False
        end = True
    elif end is True:
        label += strings[i]

result_file.close()
