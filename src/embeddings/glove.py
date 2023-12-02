import numpy as np
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors


# create GloveEmbeddings class using glove2word2vec
# see https://radimrehurek.com/gensim/scripts/glove2word2vec.html
# see

class GloVeEmbeddings:
    def __init__(self, filename):
        glove2word2vec(filename, 'glove2word2vec.txt')
        self.model = KeyedVectors.load_word2vec_format(
            'glove2word2vec.txt', binary=False)

    def __getitem__(self, word):
        return self.model[word]

    def __contains__(self, word):
        return word in self.model

    def __iter__(self):
        return iter(self.model)

    def __len__(self):
        return len(self.model)

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model = KeyedVectors.load(filename)

    def train(self, corpus, total_examples=None, epochs=None):
        pass

    def preprocess_string(self, text, language):
        pass

    def preprocess(self, corpus, language):
        pass

    def get_embedding_matrix(self, word_index, embedding_dim):
        embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
        for word, i in word_index.items():
            if word in self.model:
                embedding_matrix[i] = self.model[word]
        return embedding_matrix
