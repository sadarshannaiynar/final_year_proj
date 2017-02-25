import numpy as np
import data_helpers
import re
import h5py
from os.path import exists
from w2v import train_word2vec
from keras.models import Sequential, Model, load_model
from keras.layers import Activation, Dense, Dropout, Embedding
from keras.layers import Flatten, Input, Merge, Convolution1D, MaxPooling1D

np.random.seed(2)

sequence_length = 36
embedding_dim = 300
filter_sizes = (3, 4)
num_filters = 150
dropout_prob = (0.25, 0.5)
hidden_dims = 150
batch_size = 32
num_epochs = 5
val_split = 0.1
min_word_count = 1
context = 10

print("Loading Data...")
X, Y, vocab, vocab_inv, max_length, categories = data_helpers.load_data()
embedding_weights = train_word2vec(
    X, vocab_inv, embedding_dim, min_word_count, context)
graph_in = Input(shape=(sequence_length, embedding_dim))
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
# main sequential model
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
model.save('model')

# Training model
# ==================================================

lemma = lambda x: re.sub(r"[`?\n']", '', x.lower().strip()).split()
ques = lemma(input())
print(ques)
ques = data_helpers.get_question_vocab(ques, vocab, max_length)
print(ques)
print(categories[model.predict_classes(ques)[0]])
