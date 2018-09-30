
file1 = open('gen_paragraph__p__pre__r____final', 'r', encoding='utf-8')
file2 = open('gen_question_final', 'r', encoding='utf-8')
file3 = open('gen_label_final', 'r', encoding='utf-8')

para = file1.read().split('\a')
que = file2.read().split('\a')
la = file3.read().split('\a')

while True:
    try:
        print('입력해 ')
        idx = int(input())

        print(para[idx])
        print(que[idx])
        tk1 = la[idx].split('#')
        tk2 = para[idx].strip().split(' ')
        print(tk2[int(tk1[0]) - 1])
        print(tk1[0])

    except:
        0