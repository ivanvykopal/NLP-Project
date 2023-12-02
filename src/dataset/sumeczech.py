import pandas as pd
from .dataset import Dataset


class SumeCzechDataset(Dataset):
    def __init__(self, path=None):
        self.load_data()

    def load_data(self):
        df = pd.DataFrame()
        file_names = ['sumeczech-1.0-train.jsonl',
                      'sumeczech-1.0-dev.jsonl', 'sumeczech-1.0-test.jsonl']
        for file_name in file_names:
            df = pd.concat(
                [df, pd.read_json(f'../data/sumeczech/{file_name}', lines=True)])

        self.data = list(df['text'])
