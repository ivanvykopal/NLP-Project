from gensim.models import Word2Vec
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation, strip_tags, strip_multiple_whitespaces, strip_numeric, stem_text, strip_short, strip_short, strip_numeric, strip_punctuation, strip_tags, strip_multiple_whitespaces, stem_text, remove_stopwords
from embeddings.utils import load_stopwords
from stemmers.stemmsk import stem as stem_sk
from stemmers.stemmcz import stem_word as stem_cz
from nltk.stem import SnowballStemmer, PorterStemmer
import logging

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO
)


class Word2VecEmbeddings:
    def __init__(self, vector_size=100, window=5, min_count=3, epochs=5, sg=1, hs=0, negative=5):
        self.model = Word2Vec(vector_size=vector_size, window=window,
                              min_count=min_count, epochs=epochs, sg=sg, hs=hs, negative=negative)

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
        self.model = Word2Vec.load(filename)

    def train(self, corpus, total_examples=None, epochs=None):
        preprocessed_corpus = self.preprocess(corpus)
        logging.info('Training Word2Vec model')
        self.model.train(preprocessed_corpus,
                         total_examples=total_examples, epochs=epochs)

    def preprocess_string(self, text, language):
        stopwords = load_stopwords(language)
        if language == 'slovak':
            stem = stem_sk
        elif language == 'czech':
            stem = stem_cz
        elif language == 'english':
            stemmer = PorterStemmer()
            stem = stemmer.stem
        elif language == 'german':
            stemmer = SnowballStemmer("german")
            stem = stemmer.stem

        data = strip_tags(text)
        data = strip_punctuation(data)
        data = strip_multiple_whitespaces(data)
        data = strip_numeric(data)
        data = remove_stopwords(data, stopwords=stopwords)
        data = strip_short(data, minsize=3)
        data = stem(data)

        return data.split()

    def preprocess(self, corpus):
        return [self.preprocess_string(text) for text in corpus]
