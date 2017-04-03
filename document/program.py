import data_helpers
import numpy as np
from w2v import train_word2vec
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

embedding_dim = 300
min_word_count = 3
context = 10

print('Loading Data...')
X, Y, vocab, vocab_inv, length = data_helpers.load_data()

X = np.reshape(X, (len(X), 5, 1))
X = X / float(length)
Y = np_utils.to_categorical(Y)

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(Y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, nb_epoch=20, batch_size=128)

start = np.random.randint(0, len(X) - 1)
pattern = list(X[start])
print(pattern)
# generate characters
for i in range(1000):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(length)
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = vocab_inv[index]
    print(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]

print(X, Y)
