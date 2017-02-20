import itertools
import numpy as np
from keras.layers import Input, Embedding, merge, Flatten, SimpleRNN
from keras.models import Model
from sklearn.model_selection import train_test_split

categories = list()
target_categories = list()
question_sentences = list()
max

lemma = lambda x: x.lower().replace(
    '\n', '').replace('`', '').replace('?', '').replace('', '').strip().split()

for data in open('questions.txt', 'r'):
    data = data.split(' ', 1)
    question_sentences.append(lemma(data[1]))
    prefix = data[0].split(':')
    categories.append(prefix[0] + '_' + prefix[1])

target_categories = sorted(list(set(categories)))


def to_category_vector():
    vector = np.zeros(len(categories), dtype='int32')
    for i in range(len(categories)):
        if categories[i] == 'DESC_def':
            vector[i] = 1
    return vector

vectors = np.asarray([to_category_vector()], dtype='int32')

words = set(itertools.chain(*question_sentences))

word2idx = dict((v, i) for i, v in enumerate(words))
idx2word = list(words)

cat = open('categories.txt', 'w')

for i in target_categories:
    cat.write(i + '\n')
cat.close()

print(vectors[0].count(1))
