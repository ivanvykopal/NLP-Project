from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer


class XLMRoBERTa:
    def __init__(self, num_labels, pretrained_model_name='xlm-roberta-base'):
        super(XLMRoBERTa, self).__init__()
        self.model = XLMRobertaForSequenceClassification.from_pretrained(
            pretrained_model_name, num_labels=num_labels)
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(
            pretrained_model_name)

    def tokenize(self, text):
        return self.tokenizer(text, padding='max_length', truncation=True, return_tensors='pt')