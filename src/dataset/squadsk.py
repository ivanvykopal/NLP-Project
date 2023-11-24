import pandas as pd
from .dataset import Dataset
from datasets import load_dataset
import datasets

class SquadSKDataset(Dataset):
    def __init__(self, path) -> None:
        super().__init__(path)

    def load_data(self, path = None) -> None:
        self.data = load_dataset("TUKE-DeutscheTelekom/squad-sk")
        
        # filter only context that is unique
        data = pd.DataFrame(self.data['train'])
        data = data.drop_duplicates(subset=['context'])
        self.data = datasets.Dataset.from_pandas(data)
        self.data = self.data['context']
