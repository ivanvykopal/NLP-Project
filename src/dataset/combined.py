from .dataset import Dataset
from .slovaksum import SlovakSumDataset
from .squadsk import SquadSKDataset
from .ud import UDDataset
from .wikipedia import WikipediaDataset
from .ccnews import CCNewsDataset
from .sumeczech import SumeCzechDataset


class CombinedDataset(Dataset):
    def __init__(self, datasets=None, language=None):
        if language is None:
            self.datasets = datasets
        else:
            self.get_data_for_language(language)
        self.load_data()

    def get_data_for_language(self, language):
        if language == 'sk':
            self.datasets = [
                SlovakSumDataset(None),
                SquadSKDataset(None),
                WikipediaDataset(None, language='sk'),
                UDDataset(None, language='sk'),
            ]
        elif language == 'cs':
            self.datasets = [
                SumeCzechDataset(None),
                WikipediaDataset(None, language='cs'),
                UDDataset(None, language='cs'),
            ]
        elif language == 'en':
            self.datasets = [
                CCNewsDataset(None),
                WikipediaDataset(None, language='en'),
                UDDataset(None, language='en'),
            ]
        else:
            raise ValueError('Language not supported')
            

    def load_data(self):
        self.data = [data for dataset in self.datasets for data in dataset.get_data()]
