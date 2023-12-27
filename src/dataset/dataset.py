from tqdm import tqdm
from nltk.tokenize import sent_tokenize
import pandas as pd
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation, strip_tags, strip_multiple_whitespaces, strip_numeric, strip_short, strip_short, strip_numeric, strip_punctuation, strip_tags, strip_multiple_whitespaces, remove_stopwords
from embeddings.utils import load_stopwords
from torch.utils.data import Dataset as TorchDataset
import nltk
nltk.download('punkt')


class Dataset(TorchDataset):
    def __init__(self, path=None) -> None:
        self.data = None
        self.language = None
        if path is not None:
            self.load_data(path)
        else:
            self.load_data()

    def load_data(self, path: str = None) -> None:
        raise NotImplementedError

    def get_data(self) -> pd.DataFrame:
        return self.data

    def preprocess_string(self, text, language):
        stopwords = load_stopwords(language)

        try:
            data = text.lower()
        except:
            print(text)
            raise
        data = data.replace('„', '').replace('“', '')
        data = strip_tags(data)
        data = strip_punctuation(data)
        data = strip_multiple_whitespaces(data)
        data = strip_numeric(data)
        data = remove_stopwords(data, stopwords=stopwords)
        data = strip_short(data, minsize=3)

        return data.split()

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

    def convert_targets(self, target):
        raise NotImplementedError

    def create_vocab(self):
        vocab = sorted({
            token
            for claim in self.data['claim_tokens'].to_list()
            for token in claim
        })
        self.token2idx = {token: idx for idx, token in enumerate(vocab)}
        self.token2idx['<pad>'] = max(self.token2idx.values()) + 1
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}
        self.data['indexed_tokens'] = self.data.claim_tokens.apply(
            lambda tokens: [self.token2idx[token] for token in tokens],
        )
        self.text = self.data['claim'].to_list()
        self.sequences = self.data['indexed_tokens'].to_list()
        self.targets = self.data['label'].to_list()

    def get_stats(self):
        # add count for each class
        counts = self.data['label'].value_counts()
        return {
            'language': self.language,
            'size': len(self.sequences),
            'n_classes': len(set(self.targets)),
            'n_words': len(self.token2idx),
            'n_unique_words': len(set(self.token2idx)),
            'n_tokens': sum([len(sequence) for sequence in self.sequences]),
            'n_unique_tokens': len(set([token for sequence in self.sequences for token in sequence])),
            'class_counts': {
                'SUPPORTS': counts[1],
                'REFUTES': counts[0],
                # 'NOT ENOUGH INFO': counts[2],
            }
        }

    def print_stats(self):
        stats = self.get_stats()
        print(f'Dataset size: {stats["size"]}')
        print(f'Number of classes: {stats["n_classes"]}')
        print(f'Number of words: {stats["n_words"]}')
        print(f'Number of unique words: {stats["n_unique_words"]}')
        print(f'Number of tokens: {stats["n_tokens"]}')
        print(f'Number of unique tokens: {stats["n_unique_tokens"]}')
        print(f'Class counts: {stats["class_counts"]}')
        print(f'Language: {stats["language"]}')

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx], self.text[idx]

    def get_balanced_dataset(self):
        # remove class 2
        self.data = self.data[self.data['label'] != 2]

        # balance dataset based on the count for each class
        counts = self.data['label'].value_counts()
        min_count = min(counts)
        balanced_data = pd.concat([
            self.data[self.data['label'] == label].sample(min_count)
            for label in counts.index
        ])

        self.data = balanced_data
        self.text = self.data['claim'].to_list()
        self.sequences = self.data['indexed_tokens'].to_list()
        self.targets = self.data['label'].to_list()

    def save_data(self, path='temp.txt'):
        # save data where each sentence in one line as txt file
        sentences = [
            sent_tokenize(text.replace('\n', ''))
            for text in tqdm(self.data, desc='Tokenizing sentences')
        ]
        sentences = [item.strip() for sublist in tqdm(
            sentences, 'Splitting sentences') for item in sublist]
        # filter out empty sentences
        non_empty_sentences = [
            sentence for sentence in sentences if sentence.strip()]

        with open(path, 'w') as f:
            f.write('\n'.join(non_empty_sentences))
