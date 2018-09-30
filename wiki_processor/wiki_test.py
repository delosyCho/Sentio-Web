import xml.etree.ElementTree as ET
import numpy as np

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

                            if title == '김구라':
                                while True:
                                    print('!!')
                                    print(text.text)
                                    input()


                            if title == '#':
                                while True:
                                    print('!!')
                                    #print(text.text)
                                    TK = text.text.split('\n')
                                    input()