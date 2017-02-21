import itertools
import numpy as np
from keras.layers import Input, Embedding, merge, Flatten, SimpleRNN
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
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

vectors = np.asarray([to_category_vector()], dtype='int32').T

words = set(itertools.chain(*question_sentences))

word2idx = dict((v, i) for i, v in enumerate(words))
idx2word = list(words)

to_idx = lambda x: [word2idx[word] for word in x]
questions_idx = [to_idx(question) for question in question_sentences]
questions_idx = pad_sequences(questions_idx, maxlen=15)
questions_array = np.asarray(questions_idx, dtype='int32')

input_sentence = Input(shape=(15,), dtype='int32')
input_embedding = Embedding(len(words), 15)(input_sentence)
category_prediction = SimpleRNN(1)(input_embedding)
predict_category = Model(input=[input_sentence], output=[category_prediction])
predict_category.compile(optimizer='sgd', loss='binary_crossentropy')
predict_category.fit([questions_array], [vectors],
                     nb_epoch=5, verbose=1, batch_size=20)
embeddings = predict_category.layers[1]

print(predict_category.layers[1].W)

# for i in range(len(words)):
# print('{}: {}'.format(idx2word[i], embeddings[i]))

cat = open('categories.txt', 'w')

for i in target_categories:
    cat.write(i + '\n')
cat.close()
