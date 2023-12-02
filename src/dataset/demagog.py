import pandas as pd
from .dataset import Dataset


class DemagogDataset(Dataset):
    def __init__(self, path, language: str = 'cs') -> None:
        self.language = language
        self.load_data()

    def load_data(self, path: str = '../../data/demagog/demagog') -> None:
        path = f'{path}-{self.language}.csv'
        df = pd.read_csv(path)

        self.data = df[['claim', 'label']]
        # remove non-claim rows
        self.data = self.data[~self.data['label'].isna()]
        # remove all duplicate rows based on claim column
        self.data = self.data.drop_duplicates(subset=['claim'])

    def load_data_mixture(self, path: str = '../../data/demagog/demagog') -> None:
        languages = ['cs', 'sk']

        dfs = []
        for language in languages:
            path = f'{path}-{language}.csv'
            df = pd.read_csv(path)
            df['language'] = language
            dfs.append(df)

        df = pd.concat(dfs)
        return df[['claim', 'label', 'language']]
