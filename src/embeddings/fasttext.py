from gensim.models import FastText
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation, strip_tags, strip_multiple_whitespaces, strip_numeric, strip_short, strip_short, strip_numeric, strip_punctuation, strip_tags, strip_multiple_whitespaces, remove_stopwords
from embeddings.utils import load_stopwords
from stemmers.stemmsk import stem as stem_sk
from stemmers.stemmcz import stem_word as stem_cz
from nltk.stem import SnowballStemmer, PorterStemmer
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import nltk
nltk.download('punkt')
import logging

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO
)


class FastTextEmbeddings:
    def __init__(self, vector_size=100, window=5, min_count=3, epochs=5, sg=1, hs=0, negative=5, language='slovak'):
        logging.info('Initializing FastText model')
        self.model = FastText(vector_size=vector_size, window=window,
                              min_count=min_count, epochs=epochs, sg=sg, hs=hs, negative=negative)
        self.language = self.convert_language(language)
        logging.info('FastText model initialized')

    def convert_language(self, language):
        if language == 'sk':
            return 'slovak'
        elif language == 'cs':
            return 'czech'
        elif language == 'en':
            return 'english'
        elif language == 'de':
            return 'german'
        else:
            return language

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
        self.model = FastText.load(filename)

    def get_embed_dict(self):
        return self.model.wv

    def train(self, corpus, total_examples=None, epochs=None):
        preprocessed_corpus = self.preprocess(corpus)
        logging.info('Training FastText model')

        self.model.build_vocab(preprocessed_corpus)
        
        self.model.train(preprocessed_corpus,
                         total_examples=total_examples, epochs=epochs)

    def preprocess_string(self, text):
        stopwords = load_stopwords(self.language)
        if self.language == 'slovak':
            stem = stem_sk
        elif self.language == 'czech':
            stem = stem_cz
        elif self.language == 'english':
            stemmer = PorterStemmer()
            stem = stemmer.stem
        elif self.language == 'german':
            stemmer = SnowballStemmer("german")
            stem = stemmer.stem

        data = text.lower()
        data = data.replace('„', '').replace('“', '')
        data = strip_tags(data)
        data = strip_punctuation(data)
        data = strip_multiple_whitespaces(data)
        data = strip_numeric(data)
        data = remove_stopwords(data, stopwords=stopwords)
        data = strip_short(data, minsize=3)
        # data = stem(data)

        return data.split()

    def preprocess(self, corpus):
        logging.info('Preprocessing corpus')
        # corpus_sentences = [sent_tokenize(text) for text in tqdm(corpus)]
        # corpus_sentences = [item for sublist in tqdm(corpus_sentences) for item in sublist]
        # print(corpus_sentences[0])
        # logging.info('Tokenization')
        # return [self.preprocess_string(sentence) for sentence in tqdm(corpus_sentences)]
        return [
            self.preprocess_string(sentence) 
            for text in tqdm(corpus) 
            for sentence in sent_tokenize(text)
        ]

