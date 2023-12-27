import pandas as pd
from functools import partial
from .dataset import Dataset


class MultiClaimDataset(Dataset):
    def __init__(self, path: str = None, language: str = 'en') -> None:
        self.language = language
        self.load_data()
        self.create_vocab()

    def convert_language_triplet(self, language: str) -> str:
        if language == 'en':
            return 'eng'
        elif language == 'cs':
            return 'ces'
        elif language == 'sk':
            return 'slk'
        else:
            raise ValueError('Language not supported')
        
    def convert_targets(self, target):
        false_labels = [
            'false information', 'partly false information', 'false information.', 'lltered photo', 'partly false information.',
            'partly false', 'false', 'false information and graphic content', 'altered video', 'sensitive content',
            'altered photo/video.', 'altered photo/video', 'false headline', 'altered photo'
            ]
        true_labels = ['true', 'mostly-true', 'half-true']
        if target.strip().lower() in false_labels:
            return 0
        elif target in true_labels:
            return 1
        else:
            return 2
    

    def load_data(self, path: str = '../data/multiclaim/posts.csv') -> None:
        df = pd.read_csv(path)

        self.data = df[['text', 'verdicts']]
        # eval text and verdict
        self.data['verdicts'] = self.data['verdicts'].apply(lambda x: eval(x))
        self.data['label'] = self.data['verdicts'].apply(lambda x: x[0] if len(x) > 0 else None)

        self.data = self.data[~self.data['text'].isna()]

        self.data['text'] = self.data['text'].apply(lambda x: eval(x))
        self.data['claim'] = self.data['text'].apply(lambda x: x[0])
        self.data['language'] = self.data['text'].apply(lambda x: x[2][0][0])

        # remove columns
        self.data = self.data.drop(columns=['text', 'verdicts'])

        self.data = self.data[~self.data['label'].isna()]
        self.data.drop_duplicates(subset=['claim'], inplace=True)

        if self.language is not None:
            self.data = self.data[self.data['language'] == self.convert_language_triplet(self.language)]
            # self.data.drop(columns=['language'], inplace=True)

        self.data['claim_tokens'] = self.data.claim.apply(
            partial(self.preprocess_string, language=self.convert_language(self.language)))
        self.data = self.data[self.data['claim_tokens'].map(len) > 0]
        self.data['label'] = self.data.label.apply(self.convert_targets)
