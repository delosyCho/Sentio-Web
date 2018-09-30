# -*- coding: utf-8 -*-

from functions import *
import xml.etree.ElementTree as ET
import numpy as np

output_file = open('title_list_wiki', 'w', encoding='utf-8')

# 11개 split file
for a in range(11):
    print(a, 'processing...')

    tree = ET.parse('wikisplit' + str(a))
    root = tree.getroot()

    for page in root:
        check_ok = True

        for tags in page:
            if tags.tag == 'title':
                title = tags.text

            if tags.tag == 'revision' and check_ok is True:
                for text in tags:
                    if text.tag == 'text':
                        output_file.write(str(title) + '\a')

output_file.close()
