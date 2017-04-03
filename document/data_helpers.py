from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer, sent_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import re
import itertools
from collections import Counter

stop_words = set(stopwords.words('english'))
tokenizer = RegexpTokenizer('[\'a-zA-Z]+')
lemmatizer = WordNetLemmatizer()
lemma = lambda x: x.lower().strip().split()


def tokenize(sentence):
    tokens = [lemmatizer.lemmatize(t.lower()) for t in tokenizer.tokenize(
        sentence) if t.lower() not in stop_words]
    return tokens


def read_and_format_input_and_output_data():
    contents = [lemma(i.replace('\n', '')) for i in open(
        'computer.txt').readlines() if i.replace('\n', '')]
    contents = list(itertools.chain(*contents))
    return contents


def pad_data(contents, padding_word='<PAD/>'):
    max_length = max(len(x) for x in contents)
    padded_contents = list()
    for i in range(len(contents)):
        padded_content = contents[
            i] + [padding_word] * (max_length - len(contents[i]))
        padded_contents.append(padded_content)
    return [padded_contents, max_length]


def build_vocabulary(contents):
    word_counts = Counter(contents)
    idx_to_word = [x[0] for x in word_counts.most_common()]
    idx_to_word.append('.')
    word_to_idx = {x: i for i, x in enumerate(idx_to_word)}
    return [word_to_idx, idx_to_word]


def build_input_data(contents, vocab):
    # X = np.array([[vocab[j] for j in i] for i in contents])
    X = list()
    Y = list()
    for i in range(0, len(contents) - 5, 1):
        words_in = contents[i:i + 5]
        word_out = contents[i + 5]
        X.append([vocab[j] for j in words_in])
        Y.append(vocab[word_out])

    # Y = list()
    # for i in range(1, len(contents) - 1):
    #     Y.append([vocab[j] for j in i])
    return [X, Y]


def load_data():
    contents = read_and_format_input_and_output_data()
    # padded_contents, max_length = pad_data(contents)
    length = len(sorted(list(set(contents))))
    vocabulary, vocabulary_inv = build_vocabulary(contents)
    X, Y = build_input_data(contents, vocabulary)
    return [X, Y, vocabulary, vocabulary_inv, length]
# 9962233231
