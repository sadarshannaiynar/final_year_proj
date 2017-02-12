from gensim.models.word2vec import Word2Vec
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer, sent_tokenize
from nltk.stem import WordNetLemmatizer
from pandas import DataFrame
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM
from sklearn.cross_validation import train_test_split
import numpy as np
import random

random.seed(1000)

target_catgories = []
category_data = []

for data in open('questions.txt', 'r'):
    data = data.split(' ', 1)
    prefix = data[0].split(':')
    target_catgories.append(prefix[0] + '_' + prefix[1])

target_catgories = list(set(target_catgories))

for category in target_catgories:
    category_data.append([category, category.split('_')[0], 0])

question_categories = DataFrame(
    category_data, columns=['Type', 'Category', 'Questions'])


def update_categories(category):
    idx = question_categories[question_categories.Type == category].index[0]
    f = question_categories.get_value(idx, 'Questions')
    question_categories.set_value(idx, 'Questions', f + 1)


def to_category_vector(categories, target_catgories):
    vector = np.zeros(len(target_catgories)).astype(float)
    for i in range(len(target_catgories)):
        if target_catgories[i] in categories:
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
    question_Y[question_id] = to_category_vector(category, target_catgories)
    question_id += 1

question_categories.sort_values(by='Type', ascending=True, inplace=True)

stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer('[\'a-zA-Z]+')
lemmatizer = WordNetLemmatizer()

question_sentences = []


def tokenize(sentence):
    tokens = [lemmatizer.lemmatize(t.lower()) for t in tokenizer.tokenize(
        sentence) if t.lower() not in stop_words]
    return tokens

for key in question_X.keys():
    question_sentences.append(tokenize(question_X[key]))

word2vec_model = Word2Vec(question_sentences, size=500,
                          min_count=1, window=10, workers=5)
word2vec_model.init_sims(replace=True)
word2vec_model.save('questions.model')

num_categories = len(target_catgories)
num_of_questions = len(question_X)
X = np.zeros(shape=(num_of_questions, 20, 500)).astype(float)
Y = np.zeros(shape=(num_of_questions, num_categories)).astype(float)
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

model.add(LSTM(int(20 * 1.5), input_shape=(20, 500)))
model.add(Dropout(0.3))
model.add(Dense(num_categories))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=10,
          nb_epoch=5, validation_data=(X_test, Y_test))

accuracy = model.evaluate(X_test, Y_test, batch_size=1)

print()
print(accuracy)
