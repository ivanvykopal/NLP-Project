from transformers import AutoModel, AutoTokenizer


class XLMMLM100:
    def __init__(self, filename='xlm-mlm-100-1280'):
        self.model = AutoModel.from_pretrained(filename)
        self.tokenizer = AutoTokenizer.from_pretrained(filename)

    def tokenize(self, text):
        return self.tokenizer(text, padding='max_length', truncation=True, return_tensors='pt')

