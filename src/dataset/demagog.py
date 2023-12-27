import pandas as pd
from functools import partial
from .dataset import Dataset


class DemagogDataset(Dataset):
    def __init__(self, path: str = None, language: str = 'cs') -> None:
        self.language = language
        self.load_data()
        self.create_vocab()

    def convert_targets(self, target):
        if target in ['Nepravda', 'Zavádzajúce', 'Zavádějící']:
            return 0
        elif target in ['Pravda']:
            return 1
        elif target in ['Neoveriteľné', 'Neověřitelné']:
            return 2
        else:
            return target

    def load_data(self, path: str = '../data/demagog/demagog') -> None:
        path = f'{path}-{self.language}.csv'
        df = pd.read_csv(path)

        self.data = df[['claim', 'label']]
        # remove non-claim rows
        self.data = self.data[~self.data['label'].isna()]
        # remove all duplicate rows based on claim column
        self.data = self.data.drop_duplicates(subset=['claim'])
        self.data['claim_tokens'] = self.data.claim.apply(
            partial(self.preprocess_string, language=self.convert_language(self.language)))
        self.data = self.data[self.data['claim_tokens'].map(len) > 0]
        self.data['label'] = self.data.label.apply(self.convert_targets)

    def load_data_mixture(self, path: str = '../data/demagog/demagog') -> None:
        languages = ['cs', 'sk']

        dfs = []
        for language in languages:
            path = f'{path}-{language}.csv'
            df = pd.read_csv(path)
            df['language'] = language
            dfs.append(df)

        df = pd.concat(dfs)
        return df[['claim', 'label', 'language']]
