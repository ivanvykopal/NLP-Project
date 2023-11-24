import pandas as pd
from .dataset import Dataset


class CTKFactsDataset(Dataset):
    def __init__(self, path) -> None:
        super().__init__(path)

    def load_data(self, path: str = '../../data/ctkfacts/label_wo_delclaims.csv') -> None:
        df = pd.read_csv(path)

        self.data = df[['claim_text', 'label']]
        # rename claim_text to claim
        self.data = self.data.rename(columns={'claim_text': 'claim'})
        # remove non-claim rows
        self.data = self.data[~self.data['label'].isna()]
