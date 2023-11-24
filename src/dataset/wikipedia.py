import pandas as pd
from .dataset import Dataset
from datasets import load_dataset

class WikipediaDataset(Dataset):
    def __init__(self, path, language) -> None:
        super().__init__(path)
        self.language = language

    def load_data(self, path = None) -> None:
        if self.language == 'en':
            self.data = load_dataset("wikipedia", "20220301.en")
        elif self.language == 'sk':
            self.data = load_dataset("wikipedia", language='sk', date="20231101", beam_runner='DirectRunner')
            # remove first row from train data
            self.data['train'] = self.data['train'].iloc[1:]

        elif self.language == 'cs':
            self.data = load_dataset("wikipedia", language='cs', date="20231101", beam_runner='DirectRunner')
        else:
            raise ValueError('Language not supported')
        
        self.data = self.data['train']['text']

        