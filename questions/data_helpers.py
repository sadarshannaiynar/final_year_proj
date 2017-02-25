import numpy as np
import re
import itertools
from collections import Counter

lemma = lambda x: re.sub(r"[`?\n']", '', x.lower().strip()).split()


def get_category_vector(categories, category, flag):
    vector = np.zeros(len(categories), dtype='int32')
    for i in range(len(categories)):
        if categories[i] == category:
            vector[i] = 1
    return vector


def read_and_format_input_and_output_data():
    # desc_definition_questions = list()
    # loc_other_questions = list()
    # num_count_questions = list()
    questions = list()
    categories = [i.replace('\n', '')
                  for i in list(open('categories.txt').readlines())]
    vectors = list()
    for i in open('questions.txt').readlines():
        i = i.split(' ', 1)
        i[0] = i[0].replace(':', '_')
        questions.append(lemma(i[1]))
        vectors.append(get_category_vector(categories, i[0], False))
    #     if 'DESC:def' in i:
    #         desc_definition_questions.append(lemma(i.split(' ', 1)[1]))
    #     elif 'LOC:other' in i:
    #         loc_other_questions.append(lemma(i.split(' ', 1)[1]))
    #     elif 'NUM:count' in i:
    #         num_count_questions.append(lemma(i.split(' ', 1)[1]))
    # questions = desc_definition_questions + \
    #     loc_other_questions + num_count_questions
    # desc_labels = [[1, 0, 0] for i in desc_definition_questions]
    # loc_labels = [[0, 1, 0] for i in loc_other_questions]
    # num_labels = [[0, 0, 1] for i in num_count_questions]
    labels = np.asarray(vectors)
    return [questions, labels, categories]


def pad_questions(questions, padding_word='<PAD/>'):
    max_length = max(len(x) for x in questions)
    padded_questions = list()
    for i in range(len(questions)):
        padded_question = questions[
            i] + [padding_word] * (max_length - len(questions[i]))
        padded_questions.append(padded_question)
    return [padded_questions, max_length]


def build_vocabulary(questions):
    word_counts = Counter(itertools.chain(*questions))
    idx_to_word = [x[0] for x in word_counts.most_common()]
    word_to_idx = {x: i for i, x in enumerate(idx_to_word)}
    return [word_to_idx, idx_to_word]


def build_input_data(questions, labels, vocab):
    X = np.array([[vocab[j] for j in i] for i in questions])
    Y = np.array(labels)
    return [X, Y]


def get_question_vocab(question, vocab, max_length):
    question = question + ['<PAD/>'] * (max_length - len(question))
    X = np.array([[vocab[i] if i in vocab else len(vocab) for i in question]])
    return X


def load_data():
    questions, labels, categories = read_and_format_input_and_output_data()
    questions_padded, max_length = pad_questions(questions)
    vocab, vocab_inv = build_vocabulary(questions_padded)
    X, Y = build_input_data(questions_padded, labels, vocab)
    return [X, Y, vocab, vocab_inv, max_length, categories]
