from transformers import BertForSequenceClassification, BertTokenizer, BertConfig


class mBERT:
    def __init__(self, num_labels, pretrained_model_name='bert-base-multilingual-cased'):
        super(mBERT, self).__init__()
        self.config = BertConfig.from_pretrained(
            pretrained_model_name, num_labels=num_labels)
        self.bert = BertForSequenceClassification.from_pretrained(
            pretrained_model_name, config=self.config)
        self.tokenizer = BertTokenizer.from_pretrained(
            pretrained_model_name)

    def tokenize(self, text):
        return self.tokenizer(text, padding='max_length', truncation=True, return_tensors='pt')
