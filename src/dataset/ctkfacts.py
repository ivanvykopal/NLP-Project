import pandas as pd
from .dataset import Dataset
from functools import partial


class CTKFactsDataset(Dataset):
    def __init__(self, path=None, language: str = 'cs') -> None:
        self.language = self.convert_language(language)
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


    def load_data(self, path: str = '../../data/ctkfacts/label_wo_delclaims.csv') -> None:
        df = pd.read_csv(path)

        self.data = df[['claim_text', 'label']]
        # rename claim_text to claim
        self.data = self.data.rename(columns={'claim_text': 'claim'})
        # remove non-claim rows
        self.data = self.data[~self.data['label'].isna()]
        # drop duplicates based on claim
        self.data = self.data.drop_duplicates(subset=['claim'])
        self.data['claim_tokens'] = self.data.claim.apply(
            partial(self.preprocess_string, language=self.language))
        self.data['label'] = self.data.label.apply(self.convert_targets)
