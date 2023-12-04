import pandas as pd
from .dataset import Dataset


class MultiClaimDataset(Dataset):
    def __init__(self, path) -> None:
        super().__init__(path)

    def load_data(self, path: str = '../../data/multiclaim/posts.csv') -> None:
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
