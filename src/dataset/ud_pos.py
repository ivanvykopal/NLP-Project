import pandas as pd
from .dataset import Dataset
from conllu import parse


def flatten(sentences):
    records = []
    for sentence in sentences:
        records.append({
            'text': [token['form'] for token in sentence],
            'pos': [token['upos'] for token in sentence]
        })
    return records


class UDDatasetPOS(Dataset):
    def __init__(self, language='en', split=None) -> None:
        self.language = language
        if split is not None:
            self.splits = [split]
        else:
            self.splits = ['train', 'dev', 'test']
        self.load_data()
        self.create_vocab()

    def load_data(self) -> None:
        if self.language == 'en':
            subsets = ['en_atis', 'en_ewt', 'en_gum', 'en_lines', 'en_partut']
            df = pd.DataFrame()
            for split in self.splits:
                for subset in subsets:
                    data = parse(
                        open(f'../data/ud/{subset}-ud-{split}.conllu', 'r').read())
                    data = flatten(data)
                    data_df = pd.DataFrame(data)
                    data_df['split'] = split
                    df = pd.concat([df, data_df])

            self.data = df
        elif self.language == 'sk':
            subsets = ['sk_snk']
            df = pd.DataFrame()
            for split in self.splits:
                for subset in subsets:
                    data = parse(
                        open(f'../data/ud/{subset}-ud-{split}.conllu', 'r').read())
                    data = flatten(data)
                    data_df = pd.DataFrame(data)
                    data_df['split'] = split
                    df = pd.concat([df, data_df])
            self.data = df
        elif self.language == 'cs':
            subsets = ['cs_cac', 'cs_cltt', 'cs_fictree', 'cs_pdt']
            pdt_train = ['ca', 'ct', 'la', 'lt', 'ma', 'mt', 'va']
            df = pd.DataFrame()
            for split in self.splits:
                for subset in subsets:
                    if split == 'train' and subset == 'cs_pdt':
                        for train in pdt_train:
                            data = parse(
                                open(f'../data/ud/{subset}-ud-{split}-{train}.conllu', 'r').read())
                            data = flatten(data)
                            data_df = pd.DataFrame(data)
                            data_df['split'] = split
                            df = pd.concat([df, data_df])
                    else:
                        data = parse(
                            open(f'../data/ud/{subset}-ud-{split}.conllu', 'r').read())
                        data = flatten(data)
                        data_df = pd.DataFrame(data)
                        data_df['split'] = split
                        df = pd.concat([df, data_df])
            self.data = df
        elif self.language == 'cs_sk':
            subsets = ['cs_cac', 'cs_cltt', 'cs_fictree', 'cs_pdt', 'sk_snk']
            pdt_train = ['ca', 'ct', 'la', 'lt', 'ma', 'mt', 'va']
            df = pd.DataFrame()
            for split in self.splits:
                for subset in subsets:
                    if split == 'train' and subset == 'cs_pdt':
                        for train in pdt_train:
                            data = parse(
                                open(f'../data/ud/{subset}-ud-{split}-{train}.conllu', 'r').read())
                            data = flatten(data)
                            data_df = pd.DataFrame(data)
                            data_df['split'] = split
                            df = pd.concat([df, data_df])
                    else:
                        data = parse(
                            open(f'../data/ud/{subset}-ud-{split}.conllu', 'r').read())
                        data = flatten(data)
                        data_df = pd.DataFrame(data)
                        data_df['split'] = split
                        df = pd.concat([df, data_df])
            self.data = df
        else:
            raise ValueError('Language not supported')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx], self.text[idx]

    def create_vocab(self):
        self.vocab = set([token for sentence in self.data['text']
                          for token in sentence])
        self.vocab = sorted(list(self.vocab))
        self.vocab.append('<pad>')
        self.token2idx = {token: idx for idx,
                          token in enumerate(self.vocab)}
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}
        self.data['indexed_tokens'] = self.data.text.apply(
            lambda tokens: [self.token2idx[token] for token in tokens],
        )
        # encode also tags
        self.tags = set(
            [tag for sentence in self.data['pos'] for tag in sentence]
        )
        self.tags = sorted(list(self.tags))
        self.tags.append('<pad>')
        self.tag2idx = {tag: idx for idx, tag in enumerate(self.tags)}
        self.idx2tag = {idx: tag for tag, idx in self.tag2idx.items()}
        self.data['indexed_tags'] = self.data.pos.apply(
            lambda tags: [self.tag2idx[tag] for tag in tags],
        )

        self.sequences = self.data['indexed_tokens'].to_list()
        self.targets = self.data['indexed_tags'].to_list()
        self.text = self.data['text'].to_list()

    def get_train(self):
        # return self with train spli
        data = self.data[self.data.split == 'train']
        # return triples of indexed_tokens, indexed_tags, text
        return [
            (data.indexed_tokens.iloc[i],
             data.indexed_tags.iloc[i], data.text.iloc[i])
            for i in range(len(data))
        ]

    def get_dev(self):
        # return self with dev split
        data = self.data[self.data.split == 'dev']
        # return triples of indexed_tokens, indexed_tags, text
        return [
            (data.indexed_tokens.iloc[i],
             data.indexed_tags.iloc[i], data.text.iloc[i])
            for i in range(len(data))
        ]

    def get_test(self):
        # return self with test split
        data = self.data[self.data.split == 'test']
        # return triples of indexed_tokens, indexed_tags, text
        return [
            (data.indexed_tokens.iloc[i],
             data.indexed_tags.iloc[i], data.text.iloc[i])
            for i in range(len(data))
        ]
