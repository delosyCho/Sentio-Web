dic_file = open('master_dictionary', 'r', encoding='utf-8')
post_file = open('master_dictionary_nonepp','w', encoding='utf-8')

dics = dic_file.read().split('\n')

postposition_dic = ['은', '는', '이', '가', '을', '를', '와', '가', '에', '도', '로']

for dic in dics:
    TK = dic.split(' ')
    dic = TK[len(TK) - 1]

    if len(dic) > 1:
        check = False

        for p_dic in postposition_dic:
            if dic[len(dic) - 1] == p_dic:
                check = True

        if check is True:
            post_file.write(dic + '\n')

post_file.close()