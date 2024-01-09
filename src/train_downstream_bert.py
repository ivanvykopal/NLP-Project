import os
from downstream_task.trainer import Trainer
import argparse
from downstream_task.utils import get_config, get_wandb_config
from dataset import get_dataset
from sklearn.model_selection import train_test_split
from embeddings.fasttext import FastTextEmbeddings
from collections import namedtuple
import logging

logging.basicConfig(level=logging.INFO)

# ten random seeds
SEEDS = [42, 123, 456, 789]  # , 987, 654, 321, 111, 999, 888]


def load_embedding(path):
    embedding = FastTextEmbeddings()
    embedding.load(path)
    return embedding


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str,
                        default='./configs/config_bert.yaml')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='nlp-project')
    parser.add_argument('--sweep_config', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='liar')
    parser.add_argument('--embedding_path', type=str,
                        default='./outputs/fasttext/ud_en_embedding/ud_en_embedding.ft')
    parser.add_argument('--language', type=str, default='cs')

    # remove sweep_id.txt file
    if os.path.exists('./sweep_id.txt'):
        os.remove('./sweep_id.txt')

    args = parser.parse_args()

    wandb_config = get_wandb_config('../config/wandb.conf')
    os.environ["WANDB_API_KEY"] = wandb_config.WANDB_API_KEY
    os.environ["WANDB_USERNAME"] = wandb_config.WANDB_USERNAME
    os.environ["WANDB_DIR"] = wandb_config.WANDB_DIR
    os.environ["WANDB_PROJECT"] = args.wandb_project

    if args.sweep_config is not None:
        sweep_config = get_config(args.sweep_config)
        config = None
    else:
        config = get_config(args.config_path)
        config = namedtuple('config', config.keys())(*config.values())
        sweep_config = None

    logging.info('Loading embedding')
    embedding = load_embedding(args.embedding_path)
    embedding_dict = embedding.get_embed_dict()

    logging.info('Loading dataset')
    dataset = get_dataset(
        dataset_name=args.dataset,
        path=None,
        language=args.language
    )
    # change the dataset to have the same count for each class based on the minium count of the classes
    # dataset.get_balanced_dataset()
    dataset.create_binary_dataset()
    logging.info(f'Dataset size: {len(dataset)}')

    for seed_idx, seed in enumerate(SEEDS):
        logging.info(f'Seed: {seed}')
        # split dataset into train and validation and test
        train_data, valid_data = train_test_split(
            dataset, test_size=0.2, random_state=seed
        )
        valid_data, test_data = train_test_split(
            valid_data, test_size=0.5, random_state=seed
        )
        logging.info(f'Train size: {len(train_data)}')
        logging.info(f'Validation size: {len(valid_data)}')
        logging.info(f'Test size: {len(test_data)}')

        logging.info('Dataset stats')
        dataset.print_stats()

        trainer = Trainer(
            train_dataset=train_data,
            eval_dataset=valid_data,
            test_dataset=test_data,
            dataset=dataset,
            config=config,
            sweep_config=sweep_config,
            use_wandb=args.use_wandb,
            embedding_dict=embedding_dict,
            embedding_path=args.embedding_path,
            dataset_name=args.dataset,
            seed=seed,
        )

        logging.info('Training model')
        trainer.train_bert()

        logging.info('Evaluating model')
        trainer.evaluate_bert(config=config)