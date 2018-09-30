import numpy as np
import os
import BiCorpus_Dataset

merger = False
if merger:
    corpus_file = open('bi_lang_corpus', 'w', encoding='utf-8')
    path = 'C:\\Users\\Administrator\\Desktop\\qadataset\\Corpus10'
    file_list = os.listdir(path)
    file_list.sort()

    for f in file_list:
        print(f)
        file = open(path + '\\' + f, 'r', encoding='utf-8')
        lines = file.readlines()

        for l in lines:
            corpus_file.write(l)
            #corpus_file.write('\n')
    corpus_file.close()

    print('complete merging')
    print('press anything or stop')
    input()

dataset = BiCorpus_Dataset.BiCorpus_Dataset()

eng_file = open('bi_eng_sentences', 'w', encoding='utf-8')
kor_file = open('bi_kor_sentences', 'w', encoding='utf-8')

for i in range(len(dataset.Kor_Sentences)):
    eng_file.write(str(dataset.Eng_Sentences_[i]))
    eng_file.write('\n')

    kor_file.write(str(dataset.Kor_Sentences_[i]))
    kor_file.write('\n')

eng_file.close()
kor_file.close()

print(len(dataset.Kor_Sentences), len(dataset.Eng_Sentences))
