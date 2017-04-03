import json
from nltk import word_tokenize, pos_tag
import time

start = time.time()

print('Started: ', start)

question_json = json.load(open('train-v1.1.json'))

paragraphs = list()
questions = list()
answers = list()
for document in question_json['data']:
    for paragraph in document['paragraphs']:
        paragraphs.append(paragraph['context'])
        questions.append([qa['question'] for qa in paragraph['qas']])
        answers.append([[answer['text'] for answer in qa['answers']]
                        for qa in paragraph['qas']])

all_questions_tagged = list()
all_answers_tagged = list()
all_answers = list()

for i in range(len(paragraphs)):
    for j in range(len(questions[i])):
        tagged = list(pos_tag(word_tokenize(questions[i][j])))
        tagged = tagged[0:len(tagged) - 1]
        all_questions_tagged.append([tok_tagged[1] for tok_tagged in tagged])
        ans_tagged = list()
        for k in range(len(answers[i][j])):
            tagged = list(pos_tag(word_tokenize(answers[i][j][k])))
            tagged = tagged[0:len(tagged) - 1]
            ans_tagged.append([tok_tagged[1] for tok_tagged in tagged])
            all_answers.append(','.join([tok_tagged[1]
                                         for tok_tagged in tagged]))
        all_answers_tagged.append(ans_tagged)

print(len(all_questions_tagged))
print(len(all_answers_tagged))
print(len(all_answers))

all_unique_questions = [list(x) for x in set(tuple(x)
                                             for x in all_questions_tagged)]
all_unique_answers = list(set(all_answers))


print(len(all_unique_questions))
print(len(all_unique_answers))

end = time.time()
print('End: ', end)

print('Elapsed: ', end - start)
