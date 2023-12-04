from functools import partial
import pandas as pd
from .dataset import Dataset


class AFPDataset(Dataset):
    def __init__(self, path: str = '../data/afp/afp-all-data.csv', language: str = 'en') -> None:
        self.language = language
        self.load_data()
        self.create_vocab()

    def convert_targets(self, target):

        false_names = [
            'nepravda', 'zavádějící', 'false', 'zmanipulované video', 'zmanipulovaná fotografie', 'misleading / missing context', #cs
            'altered video', 'altered image', 'misleading', 'mostly false', 'partly false', 'altered', 'false claim', 'partially false',
            'flase', 'altered media', 'altered photo', 'falase', 'digitally-altered', 'manipulated', 'misrepresented', 'outdated', 'outdated video',
            'doctored image', 'fabricated article', 'false - satire', 'satire', 'altered image, misleading', 'false, altered news graphic',
            'photo out of context', 'misleading context', 'scam', 'fake', 'april fool', 'misrepresentation', 'photo détournée', 'fasle', 'falso', 'faux',
            'hoax', 'outdated guidance', 'out of context', 'engañoso', 'خطأ', 'manipulated image', 'satire/manipulated image',
            'zavádzajúce', 'fotomontáž', 'upravené video', 'upravená fotografia', 'zmanipulovaná fotografia', 'manipulácia', 'zmanipulovaný obrázok',
            'čiastočne nepravdivé', 'facebook posts mislead on is suspects shamima begum and lisa smith','the us embassy in ethiopia did not suggest that opposition leader will soon be released from prison',
            'these are not the words of paul kagame on his ugandan counterpart', "no arrest of tplf leaders has been made and the image shows the arrest of the former head of rwanda's presidential guard", 
            'reflections on racial divisions in society were tweeted by un representative, wrongly attributed to nelson mandela', 'these drugs are covid-19 treatments, not vaccines, and they are available in western countries',
            'cette liste contient plusieurs erreurs et approximations', "these facebook posts do not show pleas for help from sick children's families",
        ]
        true_names = ['pravda', 'true']
        not_enough_info_names = [
            'no evidence', 'missing context', 'unproven', 'unsubstantiated', 'miscaptioned', 'explainer', 'not recommended', 'misattributed',
            'unverified', 'unverified video', 'unverified image', 'unverified audio', 'unverified quote', 'unverified article', 'unverified claim',
            'study withdrawn', 'mixed', '1', '5', 'video lacks context', 'mixture', 'lacks context', 'chýba kontext', 'chýbajúci kontext', 'nepodložené',
            'chybějící kontext', 'mimo kontext', 'pfizer covid-19 vaccine ‘unapproved’ in australia?', 'vrai, mais cela se passe dans un camp aux conditions "abjectes" selon l\'onu',
            'canada funds hamas'

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

    def load_data(self, path: str = '../data/afp/afp-all-data.csv') -> None:
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
        # remove all rows, where the label is not 0, 1 or 2
        self.data = self.data[self.data['label'].isin([0, 1, 2])]

    def load_data_mixture(self, path: str = '../../data/afp/afp-all-data.csv') -> None:
        df = pd.read_csv(path)

        data = df[['claim', 'label', 'language']]
        # remove non-claim rows
        data = data[~data['label'].isna()]
        # remove all duplicate rows based on claim column
        data = data.drop_duplicates(subset=['claim'])

        return data
