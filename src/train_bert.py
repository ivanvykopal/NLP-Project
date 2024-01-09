
import argparse
import logging
from embeddings.bert_model import BertModel
from dataset import get_dataset
import nltk
import os
nltk.download('punkt')

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', help='output file')
    parser.add_argument('--dataset_name', help='dataset name')
    parser.add_argument('--dataset_path', help='dataset path', default=None)
    parser.add_argument('--language', help='language of the corpus')

    args = parser.parse_args()
    os.makedirs('./outputs/bert_model', exist_ok=True)
    final_path = os.path.join('./outputs/bert_model', args.output)
    # load wikipedia data from hugging face datasets
    logging.info('Loading dataset')
    dataset = get_dataset(args.dataset_name, args.dataset_path, args.language)
    # dataset.save_data()

    logging.info('Saving done!')
    model = BertModel()
    # model.train_tokenizer()
    logging.info('training tokenizer done!')

    model.train(epochs=1, output_dir=final_path)
