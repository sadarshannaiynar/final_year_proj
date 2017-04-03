import numpy as np
import data_helpers
import re
import h5py
import matplotlib.pyplot as plot
from os.path import exists
from w2v import train_word2vec
from keras.models import Sequential, Model, load_model
from keras.layers import Activation, Dense, Dropout, Embedding
from keras.layers import Flatten, Input, Merge, Convolution1D, MaxPooling1D
from sklearn.manifold import TSNE

np.random.seed(2)

embedding_dim = 20
filter_sizes = (3, 4)
num_filters = 150
dropout_prob = (0.25, 0.5)
hidden_dims = 150
batch_size = 128
num_epochs = 50
val_split = 0.1
min_word_count = 1
context = 10

print("Loading Data...")
X, Y, vocab, vocab_inv, max_length, categories = data_helpers.load_data()
print(len(categories))
embedding_weights = train_word2vec(
    X, vocab_inv, embedding_dim, min_word_count, context)
graph_in = Input(shape=(max_length, embedding_dim))
convs = []
for fsz in filter_sizes:
    conv = Convolution1D(nb_filter=num_filters,
                         filter_length=fsz,
                         border_mode='valid',
                         activation='relu',
                         subsample_length=1)(graph_in)
    pool = MaxPooling1D(pool_length=2)(conv)
    flatten = Flatten()(pool)
    convs.append(flatten)

if len(filter_sizes) > 1:
    out = Merge(mode='concat')(convs)
else:
    out = convs[0]

graph = Model(input=graph_in, output=out)
# if exists('model'):
#     model = load_model('model')
# else:
model = Sequential()
model.add(Embedding(len(vocab), embedding_dim, input_length=max_length,
                    weights=embedding_weights))
model.add(Dropout(dropout_prob[0], input_shape=(
    max_length, embedding_dim)))
model.add(graph)
model.add(Dense(hidden_dims))
model.add(Dropout(dropout_prob[1]))
model.add(Activation('relu'))
model.add(Dense(len(categories)))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop', metrics=['accuracy'])
model.fit(X, Y, batch_size=batch_size,
          nb_epoch=num_epochs, validation_split=val_split)
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


# lemma = lambda x: re.sub(r"[`?\n']", '', x.lower().strip()).split()
# count = 0
# print('Testing...')
# for i in open('questions_test.txt').readlines():
#     i = i.split(' ', 1)
#     i[0] = i[0].replace(':', '_')
#     ques = lemma(i[1])
#     ques = data_helpers.get_question_vocab(ques, vocab, max_length)
#     index = model.predict_classes(ques)[0]
#     if categories[index] == i[0]:
#         count += 1
#     print(i[0], i[1], categories[index])
# print(count)
# ques = lemma(input())
# print(ques)
# ques = data_helpers.get_question_vocab(ques, vocab, max_length)
# print(ques)
# print(categories[model.predict_classes(ques)[0]])
