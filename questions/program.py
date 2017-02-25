from gensim.models.word2vec import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer, sent_tokenize
from nltk.stem import WordNetLemmatizer
from pandas import DataFrame
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM
from sklearn.model_selection import train_test_split
import numpy as np
import random

random.seed(1000)

target_categories = []
category_data = []

for data in open('questions.txt', 'r'):
    data = data.split(' ', 1)
    prefix = data[0].split(':')
    target_categories.append(prefix[0] + '_' + prefix[1])

target_categories = sorted(list(set(target_categories)))

cat = open('categories.txt', 'w')

for i in target_categories:
    cat.write(i + '\n')
cat.close()

for category in target_categories:
    category_data.append([category, category.split('_')[0], 0])

question_categories = DataFrame(
    category_data, columns=['Type', 'Category', 'Questions'])


def update_categories(category):
    idx = question_categories[question_categories.Type == category].index[0]
    f = question_categories.get_value(idx, 'Questions')
    question_categories.set_value(idx, 'Questions', f + 1)


def to_category_vector(categories, target_categories):
    vector = np.zeros(len(target_categories)).astype(int)
    for i in range(len(target_categories)):
        if target_categories[i] in categories:
            vector[i] = 1.0
    return vector

question_id = 0
question_X = {}
question_Y = {}

for data in open('questions.txt', 'r'):
    data = data.split(' ', 1)
    prefix = data[0].split(':')
    category = prefix[0] + '_' + prefix[1]
    update_categories(category)
    question_X[question_id] = data[1].replace('\n', '').replace('`', '')
    question_Y[question_id] = to_category_vector(category, target_categories)
    question_id += 1

question_categories.sort_values(by='Questions', ascending=True, inplace=True)
print(question_categories)

stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer('[\'a-zA-Z]+')
lemmatizer = WordNetLemmatizer()

question_sentences = []


def tokenize(sentence):
    tokens = sentence.replace('?', '').strip().lower().split()
    return tokens

for key in question_X.keys():
    question_sentences.append(tokenize(question_X[key]))

word2vec_model = Word2Vec(question_sentences, size=500,
                          min_count=1, window=10, workers=5)
word2vec_model.save('questions.model')

num_categories = len(target_categories)
num_of_questions = len(question_X)

X = np.zeros(shape=(num_of_questions, 20, 500)).astype(float)
Y = np.zeros(shape=(num_of_questions, num_categories)).astype(int)
empty_word = np.zeros(500).astype(float)

for idx, question in enumerate(question_sentences):
    for jdx, word in enumerate(question_sentences[0]):
        if jdx == 20:
            break
        else:
            if word in word2vec_model:
                X[idx, jdx, :] = word2vec_model[word]
            else:
                X[idx, jdx, :] = empty_word

for idx, key in enumerate(question_Y.keys()):
    Y[idx, :] = question_Y[key]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

model = Sequential()

model.add(LSTM(128, input_shape=(20, 500)))
model.add(Dropout(0.3))
model.add(Dense(num_categories))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=128, nb_epoch=5)
test_question = input('Enter a question: ')
test_question_X = np.zeros(shape=(1, 20, 500)).astype(float)
for jdx, word in enumerate(test_question):
    if jdx == 20:
        break
    else:
        if word in word2vec_model:
            test_question_X[0, jdx, :] = word2vec_model[word]
        else:
            test_question_X[0, jdx, :] = empty_word
# vectors.append(model.predict_classes(test_question_X))
print(model.predict_classes(test_question_X))
# print(target_categories)
# print(list(vector).index(max(vector)))
# print(target_categories[list(vector).index(max(vector))])
