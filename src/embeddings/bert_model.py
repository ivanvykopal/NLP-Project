from transformers import BertTokenizer, LineByLineTextDataset
from transformers import BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling
import tokenizers
from transformers import Trainer, TrainingArguments


class BertModel:
    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.config = None
        self.data_collator = None

    def load_model(self):
        self.tokenizer = BertTokenizer.from_pretrained(
            './embeddings/bert_tokenizer/wikipedia_cs_sk/vocab.txt')
        self.config = BertConfig(
            vocab_size=30_522,
            hidden_size=768,
            num_hidden_layers=6,
            num_attention_heads=12,
            max_position_embeddings=512,
            do_lower_case=True,
            unk_token='[UNK]',
            sep_token='[SEP]',
            pad_token='[PAD]',
            cls_token='[CLS]',
            mask_token='[MASK]',
        )
        self.model = BertForMaskedLM(self.config)
        print('No of parameters: ', self.model.num_parameters())

        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=0.2
        )
        print('Model, Tokenizer and Config loaded successfully!')

    def save_model(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save(path)
        self.config.save_pretrained(path)

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

    def get_config(self):
        return self.config

    def train_tokenizer(self):
        bwpt = tokenizers.BertWordPieceTokenizer(None, lowercase=True)
        file_path = 'temp-wikipedia_cs_sk.txt'
        bwpt.train(
            files=[file_path],
            vocab_size=30_522,
            min_frequency=3,
            limit_alphabet=1000,
            special_tokens=[
                '[PAD]',
                '[UNK]',
                '[CLS]',
                '[SEP]',
                '[MASK]',
            ],
        )
        bwpt.save_model('./embeddings/bert_tokenizer/wikipedia_cs_sk')

    # train from scratch
    def train(self, epochs=1, batch_size=128, learning_rate=0.0005, output_dir='./embeddings/bert_model'):
        self.load_model()
        dataset = LineByLineTextDataset(
            tokenizer=self.tokenizer,
            file_path='temp-wikipedia_cs_sk.txt',
            block_size=256,
        )
        print('No. of lines: ', len(dataset))  # No of lines in your datset
        print('Sample line: ', dataset[0])

        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            save_steps=10_000,
            save_total_limit=2,
            # learning_rate=learning_rate,
            prediction_loss_only=True,
            max_steps=300_000,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=self.data_collator,
            train_dataset=dataset,
        )

        trainer.train()
        trainer.save_model(output_dir)
