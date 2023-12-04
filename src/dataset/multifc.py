import pandas as pd
from .dataset import Dataset


class MultiFCDataset(Dataset):
    def __init__(self, path) -> None:
        super().__init__(path)

    def load_data(self, path: str = '../../data/multifc/all.tsv') -> None:
        df = pd.read_csv(path, sep='\t', header=None)

        self.data = df[[1, 2]]
        # rename columns
        self.data = self.data.rename(columns={1: 'claim', 2: 'label'})

        # remove non-claim rows
        self.data = self.data[~self.data['label'].isna()]
