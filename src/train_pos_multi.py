import os
from downstream_task.trainer_pos import TrainerPOS
import argparse
from downstream_task.utils import get_config, get_wandb_config
from embeddings.fasttext import FastTextEmbeddings
from collections import namedtuple
import logging
from sklearn.model_selection import train_test_split
from dataset.ud_pos import UDDatasetPOS

logging.basicConfig(level=logging.INFO)

SEEDS = [42, 123, 456, 789, 987, 654, 321, 111, 999, 888]


def load_embedding(path):
    embedding = FastTextEmbeddings()
    embedding.load(path)
    return embedding


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str,
                        default='./configs/config.yaml')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='nlp-project')
    parser.add_argument('--sweep_config', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='ud_pos')
    parser.add_argument('--embedding_path', type=str,
                        default='./outputs/wikipedia_cs_embedding/wikipedia_cs_embedding.ft')
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
        config['embedding_path'] = args.embedding_path
        config = namedtuple('config', config.keys())(*config.values())
        sweep_config = None

    logging.info('Loading embedding')
    embedding = load_embedding(args.embedding_path)
    embedding_dict = embedding.get_embed_dict()

    logging.info('Loading dataset')
    dataset_sk = UDDatasetPOS(language='sk')
    dataset_cs = UDDatasetPOS(language='cs')
    dataset = UDDatasetPOS(language='cs_sk')

    for seed_idx, seed in enumerate(SEEDS):
        logging.info(f'Dataset size: {len(dataset)}')
        # split dataset into train and validation and test
        train_data_sk, valid_data_sk = train_test_split(
            dataset_sk, test_size=0.4, random_state=seed
        )
        valid_data_sk, test_data_sk = train_test_split(
            valid_data_sk, test_size=0.5, random_state=seed
        )

        train_data_cs, valid_data_cs = train_test_split(
            dataset_cs, test_size=0.4, random_state=seed
        )

        valid_data_cs, test_data_cs = train_test_split(
            valid_data_cs, test_size=0.5, random_state=seed
        )

        train_data = train_data_sk + train_data_cs
        valid_data = valid_data_sk + valid_data_cs
        test_data = test_data_sk

        logging.info(f'Train size: {len(train_data)}')
        logging.info(f'Validation size: {len(valid_data)}')
        logging.info(f'Test size: {len(test_data)}')

        trainer = TrainerPOS(
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
            seed=seed
        )

        logging.info('Training model')
        if args.sweep_config is not None:
            trainer.train_with_sweep()
        else:
            trainer.train()

        logging.info('Evaluating model')
        trainer.evaluate(config=config)
