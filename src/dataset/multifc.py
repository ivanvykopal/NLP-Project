import pandas as pd
from functools import partial
from .dataset import Dataset


class MultiFCDataset(Dataset):
    def __init__(self, path: str = '../data/multifc/all.tsv') -> None:
        self.language = 'en'
        self.load_data()
        self.create_vocab()

    def convert_targets(self, target):
        false_labels = [
            'pants-fire', 'false', 'barely-true', 'fiction!', 'mostly-false', 'full flop', 'pants on fire!', 'determination: misleading',
            'mostly fiction!', 'bogus warning', 'determination: false', 'incorrect', 'a lot of baloney', 'facebook scams',  'promise broken',
            'misleading', 'verdict: false', 'in-the-red', 'fiction! & satire!', 'factscan score: false', 'miscaptioned', 'scam!',
            'fake news', 'fake', 'scam', 'outdated', 'factscan score: misleading', 'some baloney', 'rating: false', 'half flip',
            'determination: huckster propaganda', 'exaggerated', 'understated', 'determination: barely true', 'outdated!', 'misattributed',
            'inaccurate attribution!', 'incorrect attribution!', 'virus!', 'spins the facts', 'disputed!', 'a little baloney', 'distorts the facts',
            'mostly false', 'cherry picks', 'misleading recommendations', 'mostly_false', 'fiction', 'conclusion: false', 'misleading!',
            'determination: a stretch', 'we rate this claim false', 'exaggerates'
        ]
        true_labels = [
            'half-true', 'mostly-true', 'true', 'determination: mostly true', 'truth!', 'promise kept', 'mostly true', 'mostly truth!',
            'confirmed authorship!', 'true messages', 'determination: true', 'verdict: true', 'authorship confirmed!', 'mostly-correct',
            'in-the-green', 'truth! & outdated!', 'correct', 'half true', 'factscan score: true', 'fact', 'previously truth! now resolved!',
            'accurate', 'mostly_true', 'correct attribution!', 'correct attribution', 'verified', 'conclusion: accurate', 'partly true',
            'partially true', 
        ]
        if target in false_labels:
            return 0
        elif target in true_labels:
            return 1
        else:
            return 2
         

    def load_data(self, path: str = '../data/multifc/all.tsv') -> None:
        df = pd.read_csv(path, sep='\t', header=None)

        self.data = df[[1, 2]]
        # rename columns
        self.data = self.data.rename(columns={1: 'claim', 2: 'label'})

        # remove non-claim rows
        self.data = self.data[~self.data['label'].isna()]
        self.data.drop_duplicates(subset=['claim'], inplace=True)
        # remove none claims
        self.data = self.data[~self.data['claim'].isna()]

        self.data['claim_tokens'] = self.data.claim.apply(
            partial(self.preprocess_string, language=self.convert_language(self.language)))
        self.data = self.data[self.data['claim_tokens'].map(len) > 0]
        self.data['label'] = self.data.label.apply(self.convert_targets)
