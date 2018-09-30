# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
from functions import *
import numpy as np

"""
1. 의
2. 도
3. 로
4. 에
5. 인
6. 린
7. 된

"""

output_file = open('eui_dictionary', 'w', encoding='utf-8')
comma = '의'

dic_list = []

# 11 split file
for a in range(11):
    print(a, 'processing...')

    tree = ET.parse('wikisplit' + str(a))
    root = tree.getroot()

    for page in root:
        title = ''

        for tags in page:
            if tags.tag == 'title':
                title = preprocess(str(tags.text)).replace('분류:', '').replace(':', '')
                TK = title.split(' ')

                if len(title) > 0:
                    if title[len(title) - 1] == comma:
                        dic_list.append(TK[len(TK) - 1])

arr = np.array(dic_list, dtype='<U20')
arr.sort()
for i in range(len(dic_list) - 1):
    if arr[i] != arr[i + 1]:
        output_file.write(arr[i])
        output_file.write('\n')
output_file.close()
