import pandas as pd
from .dataset import Dataset
from datasets import load_dataset

class SlovakSumDataset(Dataset):
    def __init__(self, path) -> None:
        super().__init__(path)

    def load_data(self, path=None) -> None:
        df = pd.DataFrame()
        data = load_dataset("kiviki/SlovakSum")
        df_data = pd.concat([pd.DataFrame(data['train']), pd.DataFrame(data['validation']), pd.DataFrame(data['test'])])
        df = pd.concat([df, df_data])
        self.data = list(df['text'])

