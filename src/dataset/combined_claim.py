import pandas as pd

from .dataset import Dataset
from .afp import AFPDataset
from .demagog import DemagogDataset
from .multiclaim import MultiClaimDataset
from .csfever import CSFEVERDataset
from .ctkfacts import CTKFactsDataset
from .fever import FEVERDataset
from .liar import LiarDataset
from .multifc import MultiFCDataset
from .xfact import XFactDataset

class CombinedClaimDataset(Dataset):
    def __init__(self, datasets=None, language=None):
        self.language = language
        if language is None:
            self.datasets = datasets
        else:
            self.get_data_for_language(language)
        self.load_data()
        self.create_vocab()

    def get_data_for_language(self, language):
        if language == 'sk':
            self.datasets = [
                AFPDataset(language='sk'),
                DemagogDataset(language='sk'),
                MultiClaimDataset(language='sk'),
            ]
        elif language == 'cs':
            self.datasets = [
                AFPDataset(language='cs'),
                CSFEVERDataset(),
                CTKFactsDataset(language='cs'),
                DemagogDataset(language='cs'),
                MultiClaimDataset(language='cs'),
            ]
        elif language == 'en':
            self.datasets = [
                AFPDataset(language='en'),
                FEVERDataset(),
                LiarDataset(),
                MultiClaimDataset(language='en'),
                MultiFCDataset(),
                XFactDataset()
            ]
        else:
            raise ValueError('Language not supported')
            

    def load_data(self):
        dfs = []
        for dataset in self.datasets:
            dfs.append(dataset.data)
        self.data = pd.concat(dfs)
        self.data.drop_duplicates(subset=['claim'], inplace=True)
        self.data.reset_index(drop=True, inplace=True)