import pandas as pd
from .dataset import Dataset
from datasets import load_dataset
import logging

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO
)

class WikipediaDataset(Dataset):
    def __init__(self, path, language) -> None:
        self.language = language
        logging.info('Loading Wikipedia dataset')
        self.load_data(path)


    def load_data(self, path = None) -> None:
        if self.language == 'en':
            self.data = load_dataset("wikipedia", "20220301.en")
        elif self.language == 'sk':
            self.data = load_dataset("wikipedia", language='sk', date="20231101", beam_runner='DirectRunner')
            # remove first row from train data
            self.data['train'] = self.data['train'][1:]

        elif self.language == 'cs':
            self.data = load_dataset("wikipedia", language='cs', date="20231101", beam_runner='DirectRunner')
        else:
            raise ValueError('Language not supported')
        
        self.data = list(self.data['train']['text'])
        logging.info('Wikipedia dataset loaded')

        