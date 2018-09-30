dics = ['# 은 ', '# 는 ', '# 이 ', '# 가 ', '# 의 ', '# 도 ']

file_name = 'corpus'
recover_input = open(file_name, 'r', encoding='utf-8')
whole_text = recover_input.read()

for dic in dics:
    whole_text = whole_text.replace(dic, '# ')

time_dics = ['년', '월', '일', '시', '분', '초']


def check_number(string):
    for a in range(len(string)):
        if not ('0' <= string[a] <= '9' or string[a] == '\n'):
            return False
    return True

result_file = open(file_name + '_', 'w', encoding='utf-8')

TK = whole_text.split(' ')
for i in range(len(TK) - 1):
    if i % 100 == 0:
        print(i, len(TK))

    if check_number(TK[i]) is True:
        is_Dic = False
        for dic in time_dics:
            if TK[i + 1] == dic:
                is_Dic = True

        if is_Dic is True:
            result_file.write(TK[i])
        else:
            result_file.write(TK[i] + ' ')
    else:
        result_file.write(TK[i] + ' ')
result_file.write(TK[len(TK) - 1])

#result_file.write(whole_text)
result_file.close()