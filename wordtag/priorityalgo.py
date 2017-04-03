import csv

f = open('wordtag.csv', 'r')
reader = csv.reader(f)
words = []
pos = []
final = {
    'n': list(),
    'v': list(),
    'j': list(),
    'i': list(),
    'm': list(),
    'd': list()
}
rule = {10: 'n', 9: 'v', 8: 'j', 7: 'i', 6: 'm', 5: 'd'}
rules = ['n', 'v', 'j', 'i', 'm', 'd']

for row in reader:
    words.append(row[0])
    pos.append(row[1])
f.close()

q = input('Type the question here..')
qwords = q.split()

for i, j in enumerate(words):
    for a in qwords:
        if a == j and pos[i] in rules:
            final[pos[i]].append(a)


print(final)
