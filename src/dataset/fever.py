import json
import pandas as pd
from functools import partial
from .dataset import Dataset


class FEVERDataset(Dataset):
    def __init__(self, path: str = '../data/fever/train.jsonl') -> None:
        self.language = 'en'
        self.load_data()
        self.create_vocab()

    def convert_targets(self, target):
        if target == 'REFUTES':
            return 0
        elif target == 'SUPPORTS':
            return 1
        else:
            return 2

    def load_data(self, path: str = '../data/fever/train.jsonl') -> None:
        with open(path, 'r') as f:
            lines = f.readlines()
            self.data = [json.loads(line) for line in lines]

        # convert self.data to pandas dataframe
        self.data = pd.DataFrame(self.data)
        # only label and claim
        self.data = self.data[['label', 'claim']]
        self.data.drop_duplicates(subset=['claim'], inplace=True)
        self.data['claim_tokens'] = self.data.claim.apply(
            partial(self.preprocess_string, language=self.convert_language(self.language)))
        self.data = self.data[self.data['claim_tokens'].map(len) > 0]
        self.data['label'] = self.data.label.apply(self.convert_targets)
