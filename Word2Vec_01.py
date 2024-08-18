from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# 示例语料
sentences = [
    "I love natural language processing",
    "Word2Vec is a powerful tool",
    "Machine learning is fascinating",
]

# 对语料进行分词
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]

# 训练 Word2Vec 模型
model = Word2Vec(tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)

# 获取词向量
word_vector = model.wv['word2vec']
print(f"Word2Vec embedding for 'word2vec':\n{word_vector}")
