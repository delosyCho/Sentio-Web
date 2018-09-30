import numpy as np

class Refined_DataProcessor:
    def __init__(self, isSQUAD=True):
        if isSQUAD is True:
            file_P = open('refined_data_paragraph', 'r', encoding='utf8')
            file_Q = open('refined_data_question', 'r', encoding='utf8')
            file_L = open('refined_data_label', 'r', encoding='utf8')
        else:
            file_P = open('WIKIQA_paragraph', 'r', encoding='utf8')
            file_Q = open('WIKIQA_question', 'r', encoding='utf8')
            file_L = open('WIKIQA_data_label', 'r', encoding='utf8')
        self.Start_Index = []
        self.Stop_Index = []
        self.Paragraphs = []
        self.Questions = []

        self.Paragraph = file_P.read().split('\a')
        self.Question = file_Q.read().split('\a')
        self.Labels = file_L.read().split('\a')

        self.Start_Index.append(0)
        for i in range(len(self.Question)):
            self.Paragraphs.append(self.Paragraph[i].split('\t'))
            self.Questions.append(self.Question[i].split('\t'))

        for i in range(1, len(self.Question) - 2):
            if self.Question[i] != self.Question[i - 1]:
                self.Start_Index.append(i)
                self.Stop_Index.append(i)
        self.Stop_Index.append(len(self.Question) - 2)

        for i in range(len(self.Start_Index)):
            for j in range(self.Start_Index[i], self.Stop_Index[i]):
                if self.Labels[j] == '':
                    print('error', j)
                    print(self.Paragraphs[j])
                    input()

        print('size:', len(self.Questions))