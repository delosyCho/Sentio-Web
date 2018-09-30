filename = 'gen_paragraph__p__pre_'
filename_ = 'gen_label_'

file = open(filename, 'r', encoding='utf-8')
file_ = open(filename_, 'r', encoding='utf-8')

result = open(filename + '_r_', 'w', encoding='utf-8')

paragraphs = file.read().replace('@ ', '@').replace(' #', '#').split('@#!')
labels = file_.read().replace('@ ', '@').replace(' #', '#').split('\a')

print(len(paragraphs))
print(len(labels))

P_Length = 200

count = 0
wrong = 0
wrong2 = 0

is_recover = True

if is_recover is True:
    for i in range(len(labels) - 1):
        if i % 500 == 0:
            print(i)

        answer = labels[i].split(',')[0].strip()
        paragraph = paragraphs[i].replace('@@' + str(i) + '##', answer)

        result.write(paragraph + '@#!')
else:
    for i in range(len(labels) - 1):
        if i % 500 == 0:
            print(i)

        answer = labels[i].split(',')[0].strip()
        paragraph = paragraphs[i].replace(answer, '@@' + str(i) + '##')

        result.write(paragraph + '@#!')
result.close()

print(count)
print(wrong)
print(wrong2)