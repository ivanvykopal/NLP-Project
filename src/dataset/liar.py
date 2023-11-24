import pandas as pd
from .dataset import Dataset


class LiarDataset(Dataset):
    def __init__(self, path) -> None:
        super().__init__(path)

    def load_data(self, path: str = '../../data/liar/train.tsv') -> None:
        df = pd.read_csv(path, sep='\t', header=None)

        self.data = df[[2, 1]]
        # rename columns
        self.data = self.data.rename(columns={2: 'claim', 1: 'label'})
        # remove non-claim rows
        self.data = self.data[~self.data['label'].isna()]
