from gensim.models import word2vec
import string

vocab_sentences = list()
train_sentences = list()
i = 0

for line in open('computers.txt'):
    vocab_sentences.append(line.replace('\n', '').split())

print('Training....')
sentences = word2vec.Text8Corpus('text8')
model = word2vec.Word2Vec.load('train.model')
model.train(vocab_sentences)
print('trained')
print(model.similarity('computer', 'device'))
# print(model.similarity('program', 'that'))
# print(model.similarity('program', 'is'))
# for i in model.vocab.keys():
# print(model[i])
# print(i)
