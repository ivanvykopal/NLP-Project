import pandas as pd
from .dataset import Dataset


class XFactDataset(Dataset):
    def __init__(self, path) -> None:
        super().__init__(path)

    def load_data(self, path: str = '../../data/x-fact/train.all.tsv') -> None:
        df = pd.read_csv(path, sep='\t', encoding='utf-8', on_bad_lines='skip')

        self.data = df[['claim', 'label', 'language']]
        # remove non-claim rows
        self.data = self.data[~self.data['label'].isna()]
