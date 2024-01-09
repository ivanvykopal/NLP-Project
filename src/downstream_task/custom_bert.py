from transformers import BertForSequenceClassification, BertTokenizer
import torch


class CustomBERT:
    def __init__(self, config) -> None:
        self.model_path = config.model_path
        self.tokenizer_path = config.tokenizer_path
        self.model = None
        self.tokenizer = None
        self.config = config
        self.load_model()

    def load_model(self):
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_path,
            num_labels=self.config.output_dim,
            output_attentions=False,
            output_hidden_states=False
        )

        for param in self.model.bert.parameters():
            param.requires_grad = False

        self.tokenizer = BertTokenizer.from_pretrained(
            self.tokenizer_path, do_lower_case=True)

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

    def forward(self, input_ids, attention_mask, labels=None, token_type_ids=None):
        return self.model(input_ids, attention_mask, labels=labels, token_type_ids=token_type_ids)

    def preprocess_data(self, dataset):

        claims = [claim for _, _, claim in dataset]
        labels = [target for _, target, _ in dataset]
        max_length = 512

        input_ids = []
        attention_masks = []

        sentence_ids = []
        counter = 0

        for claim in claims:
            encoded_dict = self.tokenizer.encode_plus(
                str(claim),
                add_special_tokens=True,
                max_length=max_length,
                pad_to_max_length=True,
                return_attention_mask=True,
                truncation=True,
                return_tensors='pt'
            )

            input_ids.append(encoded_dict['input_ids'])
            attention_masks.append(encoded_dict['attention_mask'])

            sentence_ids.append(counter)
            counter += 1

        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.tensor(labels)
        sentence_ids = torch.tensor(sentence_ids)

        return input_ids, attention_masks, labels
