# Train custom embeddings using FastText
#
# Usage: python train_fasttext.py <corpus> <output>
#
# <corpus> is a text file with one sentence per line
# <output> is the filename to save the model to
#
# Example: python train_fasttext.py data/corpus.txt data/embeddings/fasttext.model
#
# The model will be saved in the gensim format, which is a directory containing
# several files. The most important ones are:
# * model: the trained embeddings
# * model.trainables.syn1neg.npy: the negative sampling weights
# * model.wv.vectors.npy: the word vectors

import argparse
import logging
from embeddings.fasttext import FastTextEmbeddings
from dataset import get_dataset

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', help='output file')
    parser.add_argument('--dataset_name', help='dataset name')
    parser.add_argument('--dataset_path', help='dataset path', default=None)
    parser.add_argument('--language', help='language of the corpus')

    args = parser.parse_args()
    # load wikipedia data from hugging face datasets
    logging.info('Loading dataset')
    dataset = get_dataset(args.dataset_name, args.dataset_path, args.language)
    data = dataset.get_data()

    # train embeddings
    embeddings = FastTextEmbeddings(language=args.language)
    logging.info('Training FastText model')

    embeddings.train(list(data), total_examples=len(list(data)), epochs=5)

    # save embeddings
    logging.info('Saving model to {}'.format(args.output))
    embeddings.save(args.output)
