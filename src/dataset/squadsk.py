import pandas as pd
from .dataset import Dataset
from datasets import load_dataset
import datasets

class SquadSKDataset(Dataset):
    def __init__(self, path) -> None:
        super().__init__(path)

    def load_data(self, path = None) -> None:
        data = load_dataset("TUKE-DeutscheTelekom/squad-sk")
        df = pd.DataFrame()
        df_data = pd.concat([pd.DataFrame(data['train']), pd.DataFrame(data['validation'])])
        df = pd.concat([df, df_data])
        # filter only context that is unique
        df = df.drop_duplicates(subset=['context'])
        self.data = list(df['context'])
