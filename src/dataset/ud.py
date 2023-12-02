import pandas as pd
from .dataset import Dataset
from datasets import load_dataset


class UDDataset(Dataset):
    def __init__(self, path, language) -> None:
        self.language = language
        self.load_data(path)
    
    def load_data(self, path = None) -> None:
        if self.language == 'en':
            subsets = ['en_esl', 'en_ewt', 'en_gum', 'en_gumreddit', 'en_lines', 'en_partut']
            df = pd.DataFrame()
            for subset in subsets:
                data = load_dataset('albertvillanova/universal_dependencies', subset)
                # combine all splits train, validation and test
                df_data = pd.concat([pd.DataFrame(data['train']), pd.DataFrame(data['validation']), pd.DataFrame(data['test'])])
                df = pd.concat([df, df_data])
            self.data = df
        elif self.language == 'sk':
            data = load_dataset('albertvillanova/universal_dependencies', 'sk_snk')
            # combine all splits train, validation and test
            self.data = pd.concat([pd.DataFrame(data['train']), pd.DataFrame(data['validation']), pd.DataFrame(data['test'])])
        elif self.language == 'cs':
            # load cs_cac, cs_cltt, cs_fictree, cs_pdt, cs_pud,
            subsets = ['cs_cac', 'cs_cltt', 'cs_fictree', 'cs_pdt']
            df = pd.DataFrame()
            for subset in subsets:
                data = load_dataset('albertvillanova/universal_dependencies', subset)
                # combine all splits train, validation and test
                df_data = pd.concat([pd.DataFrame(data['train']), pd.DataFrame(data['validation']), pd.DataFrame(data['test'])])
                df = pd.concat([df, df_data])
            self.data = df
        else:
          raise ValueError('Language not supported')
        
        self.data = list(self.data['text'])