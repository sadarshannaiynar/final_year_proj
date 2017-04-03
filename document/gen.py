from gensim.summarization import summarize, keywords

content = open('computer.txt').read()

content = open('../questions/questions_test.txt').read()
# print(content)
print(keywords(content))
