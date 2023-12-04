import pandas as pd
from functools import partial
from .dataset import Dataset


class LiarDataset(Dataset):
    def __init__(self, path: str = '../data/liar/train.tsv') -> None:
        self.language = self.convert_language('en')
        self.load_data()

    def convert_targets(self, target):
        # ['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']
        if target in ['pants-fire', 'false', 'barely-true']:
            return 0
        elif target in ['half-true', 'mostly-true', 'true']:
            return 1
        else:
            return 2

    def load_data(self, path: str = '../data/liar/train.tsv') -> None:
        df = pd.read_csv(path, sep='\t', header=None)

        self.data = df[[2, 1]]
        # rename columns
        self.data = self.data.rename(columns={2: 'claim', 1: 'label'})
        # remove non-claim rows
        self.data = self.data[~self.data['label'].isna()]
        self.data.drop_duplicates(subset=['claim'], inplace=True)
        self.data['claim_tokens'] = self.data.claim.apply(
            partial(self.preprocess_string, language=self.language))
        self.data['label'] = self.data.label.apply(self.convert_targets)