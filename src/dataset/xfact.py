import pandas as pd
from functools import partial
from .dataset import Dataset


class XFactDataset(Dataset):
    def __init__(self, path: str = '../data/x-fact/train.all.tsv', language='en') -> None:
        self.language = language
        self.load_data()
        self.create_vocab()

    def convert_targets(self, target):
        if target in ['mostly false', 'false']:
            return 0
        elif target in ['mostly true', 'true', 'half true']:
            return 1
        else:
            return 2

    def load_data(self, path: str = '../data/x-fact/train.all.tsv') -> None:
        df = pd.read_csv(path, sep='\t', encoding='utf-8', on_bad_lines='skip')

        self.data = df[['claim', 'label', 'language']]
        # remove non-claim rows
        self.data = self.data[~self.data['label'].isna()]
        self.data.drop_duplicates(subset=['claim'], inplace=True)
        self.data = self.data[self.data['language'] == self.language]

        self.data['claim_tokens'] = self.data.claim.apply(
            partial(self.preprocess_string, language=self.convert_language(self.language)))
        self.data = self.data[self.data['claim_tokens'].map(len) > 0]
        self.data['label'] = self.data.label.apply(self.convert_targets)
