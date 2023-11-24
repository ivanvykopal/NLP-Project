import json
import pandas as pd
from .dataset import Dataset


class FEVERDataset(Dataset):
    def __init__(self, path) -> None:
        super().__init__(path)

    def load_data(self, path: str = '../../data/fever/train.jsonl') -> None:
        with open(path, 'r') as f:
            lines = f.readlines()
            self.data = [json.loads(line) for line in lines]

        # convert self.data to pandas dataframe
        self.data = pd.DataFrame(self.data)
        # only label and claim
        self.data = self.data[['label', 'claim']]
