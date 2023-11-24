import pandas as pd
from .dataset import Dataset
from datasets import load_dataset

class SlovakSumDataset(Dataset):
    def __init__(self, path) -> None:
        super().__init__(path)

    def load_data(self, path) -> None:
        self.data = load_dataset("kiviki/SlovakSum")
        self.data = self.data['train']['text']
        self.data = pd.DataFrame(self.data)
