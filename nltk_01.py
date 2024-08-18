import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt_tab')
text = "NLTK is a powerful library for natural language processing. It is widely used in academic research."
# 句子分割
sentences = sent_tokenize(text)
print(sentences)

# 词分割
words = word_tokenize(text)
print(words)
