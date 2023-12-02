from functools import partial
import pandas as pd
from .dataset import Dataset


class AFPDataset(Dataset):
    def __init__(self, path: str, language: str = 'en') -> None:
        self.language = language
        self.load_data()
        self.create_vocab()

    def convert_targets(self, target):

        false_names = [
            'nepravda', 'zavádějící', 'false', 'zmanipulované video', 'zmanipulovaná fotografie', 'nisleading / nissing context', #cs
            'altered video', 'altered image', 'misleading', 'mostly false', 'partly false', 'altered', 'false claim', 'partially false',
            'flase', 'altered media', 'altered photo', 'falase', 'digitally-altered', 'manipulated', 'misrepresented', 'outdated', 'outdated video',
            'doctored image', 'fabricated article', 'false - satire', 'satire', 'altered image, misleading', 'false, altered news graphic',
            'photo out of context', 'misleading context', 'scam', 'fake', 'april fool', 'misrepresentation', 'photo détournée', 'fasle', 'falso', 'faux',
            'hoax', 'outdated guidance', 'out of context', 'Eegañoso', 'خطأ', 'manipulated image', 'satire/manipulated image'
        ]
        true_names = ['pravda', 'true']
        not_enough_info_names = [
            'no evidence', 'missing context', 'unproven', 'unsubstantiated', 'miscaptioned', 'explainer', 'not recommended', 'misattributed',
            'unverified', 'unverified video', 'unverified image', 'unverified audio', 'unverified quote', 'unverified article', 'unverified claim',
            'study withdrawn', 'mixed', '1', '5', 'video lacks context', 'mixture', 'lacks context'
        ]

        if target.lower() in false_names:
            return 0
        elif target.lower() in true_names:
            return 1
        elif target.lower() in not_enough_info_names:
            return 2
        elif 'false' in target.lower() or 'misleading' in target.lower() or 'No,' in target:
            return 0
        elif 'no evidence' in target.lower():
            return 2
        return target

    def load_data(self, path: str = '../../data/afp/afp-all-data.csv') -> None:
        df = pd.read_csv(path)

        self.data = df[['claim', 'label', 'language']]
        # remove non-claim rows
        self.data = self.data[~self.data['label'].isna()]
        # print rows with nan claim
        self.data = self.data[~self.data['claim'].isna()]
        # remove all duplicate rows based on claim column
        self.data = self.data.drop_duplicates(subset=['claim'])

        self.data = self.data[self.data['language'] == self.language]
        self.data['claim_tokens'] = self.data.claim.apply(
            partial(self.preprocess_string, language=self.convert_language(self.language)))
        self.data['label'] = self.data.label.apply(self.convert_targets)

    def load_data_mixture(self, path: str = '../../data/afp/afp-all-data.csv') -> None:
        df = pd.read_csv(path)

        data = df[['claim', 'label', 'language']]
        # remove non-claim rows
        data = data[~data['label'].isna()]
        # remove all duplicate rows based on claim column
        data = data.drop_duplicates(subset=['claim'])

        return data
