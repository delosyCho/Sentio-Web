import numpy as np

class BiCorpus_Dataset:

    def Trim_Sentence(self, string, line_num):
        #print(string)

        TK = string[1:len(string)]

        TK = TK.replace('.', '')
        TK = TK.replace('!', '')
        TK = TK.replace('?', '')

        for k in range(3):
            try:
                if TK[len(TK) - 1] == ' ':
                    TK = TK[0:len(TK) - 1]

            except:
                print('!!!')
                print(string)
                print('!!!', line_num)
                input()

        return TK

    def __init__(self):

        """
        word2vec kor
        """

        """
        word2vec eng
        """

        """
        dataset
        """
        self.Kor_Sentences = []
        self.Eng_Sentences = []

        self.Kor_Sentences_ = []
        self.Eng_Sentences_ = []

        file = open('bi_lang_corpus', 'r', encoding='utf-8')
        lines = file.read().split('\n')
        line_count = 0

        is_Eng = True

        while True:
            if line_count >= len(lines) - 1:
                break

            """
            print(lines[line_count])
            print(lines[line_count + 1])
            print(lines[line_count + 2])
            print('------------------', line_count, '/', len(lines))
            input()
            """

            if lines[line_count] == '':
                line_count += 1
            elif lines[line_count] == '#...':
                line_count += 2
            elif lines[line_count][0] == '':
                line_count += 1
            elif lines[line_count][0] == '[':
                line_count += 1
            elif lines[line_count][0] == '#':

                is_okay = True
                if lines[line_count + 1] != '':
                    if lines[line_count + 1] == '#' or lines[line_count + 1] == '# ':
                        is_okay = False

                if is_okay:
                    TK = str(self.Trim_Sentence(lines[line_count], line_count))
                    if is_Eng:
                        is_Eng = False
                        self.Eng_Sentences.append(TK.split())
                        self.Eng_Sentences_.append(TK)

                        line_count += 1
                        #print('!!!!!!')
                    else:
                        is_Eng = True
                        self.Kor_Sentences.append(TK.split())
                        self.Kor_Sentences_.append(TK)

                        line_count += 1
                else:
                    line_count += 2
                    #print('@@@@@@')

            else:
                line_count += 1





