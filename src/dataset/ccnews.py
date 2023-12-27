import pandas as pd
from .dataset import Dataset
from datasets import load_dataset
import logging

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO
)


class CCNewsDataset(Dataset):
    def __init__(self, path) -> None:
        logging.info('Loading CC News dataset')
        self.load_data(path)

    def load_data(self, path=None) -> None:

        self.data = load_dataset("cc_news")

        self.data = self.data['train']['text']
        logging.info('CC News dataset loaded')
