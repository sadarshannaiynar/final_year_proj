import json
import rake
import re
import numpy as np
from keras.models import model_from_json

vocab = json.load(open('../questions/vocab.json'))
vocab_inv = json.load(open('../questions/vocab_inv.json'))
question_json = json.load(open('dev-v1.1.json'))
max_length = vocab["max_length"]

categories = [i for i in open('../questions/categories.txt').readlines()]

lemma = lambda x: re.sub(r"[`?\n']", '', x.lower().strip()).split()


def get_question_vocab(question, vocab, max_length):
    question = lemma(question)
    question = question + ['<PAD/>'] * (max_length - len(question))
    X = np.array([[vocab[i] if i in vocab else vocab['<OTHER/>']
                   for i in question]])
    return X

paragraphs = list()
questions = list()
answers = list()
for document in question_json['data']:
    for paragraph in document['paragraphs']:
        paragraphs.append(paragraph['context'])
        questions.append([qa['question'] for qa in paragraph['qas']])
        answers.append([[answer['text'] for answer in qa['answers']]
                        for qa in paragraph['qas']])
        break
    break

json_file = open('../questions/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights('../questions/model.h5')
print('Loaded model from disk')
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop', metrics=['accuracy'])

for i in range(len(paragraphs)):
    words = list()
    for j in range(len(questions[i])):
        rake_object = rake.Rake("Stoplist.txt", 0, 10, 0)
        keywords = rake_object.run(questions[i][j])
        for keyword in keywords:
            words.append(keyword[0])
        ques = get_question_vocab(questions[i][j], vocab, max_length)
        print(questions[i][j], categories[model.predict_classes(ques)[0]])
    words = list(set(words))
print(len(questions[0]))
