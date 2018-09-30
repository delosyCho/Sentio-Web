import QA_Data_Generator

file = open('gen_label', 'r', encoding='utf-8')
file1 = open('gen_paragraph_pre', 'r', encoding='utf-8')
file2 = open('gen_question', 'r', encoding='utf-8')

print(len(file.read().split('\a')))
print(len(file1.read().split('@#!')))
print(len(file2.read().split('\a')))

input()
generator = QA_Data_Generator.Data_Generator()