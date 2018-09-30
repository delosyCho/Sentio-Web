f = open('wiki_corpus', 'r', encoding='utf-8')
f2 = open('wiki_corpus_little', 'w', encoding='utf-8')

lines = f.readlines()
print(len(lines))

for i in range(20000):
    f2.write(lines[i])
f2.close()