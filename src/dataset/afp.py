import pandas as pd
from .dataset import Dataset


class AFPDataset(Dataset):
    def __init__(self, path: str, language: str = 'en') -> None:
        super().__init__(path)
        self.load_data(path)
        self.language = language

    def load_data(self, path: str = '../../data/afp/afp-all-data.csv') -> None:
        df = pd.read_csv(path)

        self.data = df[['claim', 'label', 'language']]
        # remove non-claim rows
        self.data = self.data[~self.data['label'].isna()]
        # remove all duplicate rows based on claim column
        self.data = self.data.drop_duplicates(subset=['claim'])

        self.data = self.data[self.data['language'] == self.language]

    def load_data_mixture(self, path: str = '../../data/afp/afp-all-data.csv') -> None:
        df = pd.read_csv(path)

        data = df[['claim', 'label', 'language']]
        # remove non-claim rows
        data = data[~data['label'].isna()]
        # remove all duplicate rows based on claim column
        data = data.drop_duplicates(subset=['claim'])

        return data
