import numpy as np
from gensim.models import KeyedVectors
from gensim.models import Word2Vec as word2vec
from keras.models import load_model

text = "Đồ ăn ngon nhất hà nội luôn."
model_sentiment = load_model("./models.h5")
model_embedding = KeyedVectors.load('./word_embedding.model')
max_seq = 200
embedding_size = 128
word_labels = []
for word in model_embedding.vocab.keys():
    word_labels.append(word)
    
def pre_process(text):
    punc = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''
    for ele in text: 
        if ele in punc: 
            test_str = text.replace(ele, "") 
    text = text.strip(' ').lower()
    return text

def comment_embedding(comment):
    matrix = np.zeros((max_seq, embedding_size))
    words = comment.split()
    lencmt = len(words)

    for i in range(max_seq):
        indexword = i % lencmt
        if (max_seq - i < lencmt):
            break
        if(words[indexword] in word_labels):
            matrix[i] = model_embedding[words[indexword]]
    matrix = np.array(matrix)
    return matrix

if __name__ == '__main__':
    text = pre_process(text)

    maxtrix_embedding = np.expand_dims(comment_embedding(text), axis=0)
    maxtrix_embedding = np.expand_dims(maxtrix_embedding, axis=3)
    result = model_sentiment.predict(maxtrix_embedding)
    result = result.squeeze()
    top_class = np.argmax(result)
    if top_class ==0:
        type_cmt = 'Positive'
    else:
        type_cmt = 'Negative'
    print("Label predict: {}, Sore: {}".format(type_cmt,result[top_class]))
