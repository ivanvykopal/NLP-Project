import json
import pandas as pd
from functools import partial
from .dataset import Dataset


class CSFEVERDataset(Dataset):
    def __init__(self, path: str = '../data/fever/train_cs.jsonl') -> None:
        self.language = self.convert_language('cs')
        self.load_data()
        self.create_vocab()

    def convert_targets(self, target):
        # ['SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO']
        if target == 'REFUTES':
            return 0
        elif target == 'SUPPORTS':
            return 1
        else:
            return 2

    def load_data(self, path: str = '../data/fever/') -> None:
        files = ['train_cs.jsonl', 'dev_cs.jsonl']
        self.data = []

        for file in files:
            cur_path = path + file
            with open(cur_path, 'r') as f:
                lines = f.readlines()
                data = [json.loads(line) for line in lines]
                self.data.extend(data)

        # convert self.data to pandas dataframe
        self.data = pd.DataFrame(self.data)
        # only label and claim
        self.data = self.data[['label', 'claim']]
        self.data.drop_duplicates(subset=['claim'], inplace=True)
        self.data['claim_tokens'] = self.data.claim.apply(
            partial(self.preprocess_string, language=self.language))
        self.data = self.data[self.data['claim_tokens'].map(len) > 0]
        self.data['label'] = self.data.label.apply(self.convert_targets)
