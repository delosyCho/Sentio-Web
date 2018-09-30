import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QAction, QLineEdit, QMessageBox
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
from utils import *
import random

from PyQt5.QtWidgets import QMessageBox

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'article body extractor'
        self.left = 100
        self.top = 100
        self.width = 700
        self.height = 220
        self.initUI()

        self.mode = 0
        self.crawler = Web_Crawler()

        self.rule_type = []
        self.urls = []
        self.div_tags = []


    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # Create textbox
        self.textbox = QLineEdit(self)
        self.textbox.move(20, 10)
        self.textbox.resize(400, 20)
        self.textbox.setText('Url을 입력하시오')

        self.textbox3 = QLineEdit(self)
        self.textbox3.move(20, 40)
        self.textbox3.resize(400, 20)
        self.textbox3.setText('탐색하고자 하는 텍스트, 페이지이름, 사전이름, 규칙 등을 입력하세요')

        self.textbox2 = QLineEdit(self)
        self.textbox2.move(20, 70)
        self.textbox2.resize(400, 20)
        self.textbox2.setText('규칙 이름, 사전내용')

        # Create a button in the window
        self.button = QPushButton('Reset_file', self)
        self.button.move(0, 100)

        self.button2 = QPushButton('article body', self)
        self.button2.move(120, 100)

        self.button3 = QPushButton('href tag', self)
        self.button3.move(240, 100)

        self.button4 = QPushButton('page name', self)
        self.button4.move(0, 150)

        self.button5 = QPushButton('rule', self)
        self.button5.move(120, 150)

        self.button6 = QPushButton('dictionary', self)
        self.button6.move(240, 150)

        self.button7 = QPushButton('pref', self)
        self.button7.move(0, 190)

        self.button8 = QPushButton('main', self)
        self.button8.move(120, 190)

        # connect button to function on_click
        self.button.clicked.connect(self.on_click)
        self.button2.clicked.connect(self.on_click2)
        self.button3.clicked.connect(self.on_click3)
        self.button4.clicked.connect(self.on_click4)
        self.button5.clicked.connect(self.on_click5)
        self.button6.clicked.connect(self.on_click6)
        self.button7.clicked.connect(self.on_click7)
        self.button8.clicked.connect(self.on_click8)

        self.show()

    @pyqtSlot()
    def on_click(self):
        textboxValue = self.textbox.text()
        print(textboxValue)
        self.textbox.setText("")

        file_name = 'data_file'
        file = open(file_name, 'w', encoding='utf-8')
        file.close()

    def on_click2(self):
        self.title = 'article body extractor'
        self.setWindowTitle(self.title)

        url = self.textbox.text()
        div_text = self.textbox3.text()
        original_url = url.split('/')[2]

        self.crawler.get_div_list(url=url)
        selected_tag, selected_id = self.crawler.get_most_contain_tag(url=url, text=div_text)

        file_name = 'data_file'
        try:
            file = open(file_name, 'a', encoding='utf-8')
        except:
            file = open(file_name, 'w', encoding='utf-8')
        file.write('0#' + original_url + '#' + selected_tag + '#' + selected_id + '\n')
        file.close()
        QMessageBox.about(self, "Parsing", "Complete")

    def on_click3(self):
        self.title = 'article body extractor'
        self.setWindowTitle(self.title)

        url = self.textbox.text()
        div_text = self.textbox3.text()
        original_url = url.split('/')[2]

        self.crawler.get_div_list(url=url)
        selected_tag, selected_id = self.crawler.get_most_contain_tag(url=url, text=div_text)

        file_name = 'data_file'
        try:
            file = open(file_name, 'a', encoding='utf-8')
        except:
            file = open(file_name, 'w', encoding='utf-8')
        file.write('1#' + original_url + '#' + selected_tag + '#' + selected_id + '\n')
        file.close()

        QMessageBox.about(self, "Parsing", "Complete")

    def on_click4(self):
        self.title = 'page name'
        self.setWindowTitle(self.title)

        url = self.textbox.text()
        page_name = self.textbox3.text()
        original_url = url.split('/')[2]

        file_name = 'data_file'
        try:
            file = open(file_name, 'a', encoding='utf-8')
        except:
            file = open(file_name, 'w', encoding='utf-8')
        file.write('2#' + original_url + '#' + page_name + '\n')
        file.close()

        QMessageBox.about(self, "Parsing", "Complete")

    def on_click5(self):
        self.title = 'rule edit'
        self.setWindowTitle(self.title)

        url = self.textbox.text()
        rule = self.textbox3.text()
        rule_names = self.textbox2.text()

        original_url = url.split('/')[2]

        file_name = 'data_file'
        try:
            file = open(file_name, 'a', encoding='utf-8')
        except:
            file = open(file_name, 'w', encoding='utf-8')
        file.write('3#' + original_url + '#' + rule + '#' + rule_names + '\n')
        file.close()

        QMessageBox.about(self, "Parsing", "Complete")

    def on_click6(self):
        self.title = 'dictionary'
        self.setWindowTitle(self.title)

        dic_value = self.textbox2.text()
        dic_name = self.textbox3.text()

        file_name = 'data_file'
        try:
            file = open(file_name, 'a', encoding='utf-8')
        except:
            file = open(file_name, 'w', encoding='utf-8')
        file.write('4#' + dic_name + '#' + dic_value + '\n')
        file.close()

        QMessageBox.about(self, "Parsing", "Complete")

    def on_click7(self):
        self.title = 'preference'
        self.setWindowTitle(self.title)

        url = self.textbox.text()
        pref_text = self.textbox3.text()
        pref_name = self.textbox2.text()

        self.crawler.get_div_list(url=url)
        selected_tag, selected_id = self.crawler.get_most_contain_tag(url=url, text=pref_text)

        file_name = 'data_file'
        try:
            file = open(file_name, 'a', encoding='utf-8')
        except:
            file = open(file_name, 'w', encoding='utf-8')
        file.write('5#' + url + '#' + pref_name + '#' + selected_tag + '#' + selected_id + '\n')
        file.close()

        QMessageBox.about(self, "Parsing", "Complete")

    def on_click8(self):
        self.title = 'main'
        self.setWindowTitle(self.title)

        url = self.textbox.text()
        pref_text = self.textbox3.text()
        original_url = url.split('/')[2]

        self.crawler.get_div_list(url=url)
        selected_tag, selected_id = self.crawler.get_most_contain_tag(url=url, text=pref_text)

        file_name = 'data_file'
        try:
            file = open(file_name, 'a', encoding='utf-8')
        except:
            file = open(file_name, 'w', encoding='utf-8')
        file.write('6#' + original_url + '#' + selected_tag + '#' + selected_id + '\n')
        file.close()

        QMessageBox.about(self, "Parsing", "Complete")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())