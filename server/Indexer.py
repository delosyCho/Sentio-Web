import numpy as np
import Dictionary

class Indexer:
    def __init__(self):
        title_file = open('title_list_wiki', 'r', encoding='utf-8')
        corpus_file = open('wiki_corpus', 'r', encoding='utf-8')

        titles = title_file.read().split('\a')
        documents = corpus_file.read().split('\a')

        while(True):
            index = int(input())
            print(titles[index])
            print(documents[index])

        print(len(titles))
        print(len(documents))
        input()

        self.Dictionary = Dictionary.Dictionary()

        #데이터 파일과 Indexing File들이 저장 될 경로를 지정
        self.path = ''

        file = open(self.path, 'r', encoding='utf-8')
        whole_text = file.read()
        documents = whole_text.split('\t')

        self.number_of_document = len(documents)

        for i in range(self.number_of_document):
            0

indexer = Indexer()