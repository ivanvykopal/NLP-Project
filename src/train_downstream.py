import os
from downstream_task.trainer import Trainer
import argparse
import torch
from torch.utils.data import DataLoader
from downstream_task.utils import get_config, get_model, get_wandb_config, get_metrics
# from downstream_task.metrics import get_metrics
from dataset import get_dataset
from sklearn.model_selection import train_test_split
from embeddings.fasttext import FastTextEmbeddings
from collections import namedtuple
import logging

logging.basicConfig(level=logging.INFO)

def load_embedding(path):
    embedding = FastTextEmbeddings()
    embedding.load(path)
    return embedding


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./configs/config.yaml')
    parser.add_argument('--use_wandb', action='store_true')

    args = parser.parse_args()

    wandb_config = get_wandb_config('../config/wandb.conf')
    os.environ["WANDB_API_KEY"] = wandb_config.WANDB_API_KEY
    os.environ["WANDB_USERNAME"] = wandb_config.WANDB_USERNAME
    os.environ["WANDB_DIR"] = wandb_config.WANDB_DIR

    config = get_config(args.config_path)
    config = namedtuple('config', config.keys())(*config.values())

    logging.info('Loading embedding')
    embedding = load_embedding(config.embedding_path)
    embedding_dict = embedding.get_embed_dict()

    logging.info('Loading dataset')
    dataset = get_dataset(
        dataset_name=config.dataset,
        path=None,
        language=config.language
    )

    logging.info(f'Dataset size: {len(dataset)}')

    # split dataset into train and validation and test
    train_data, valid_data = train_test_split(
        dataset, test_size=0.4, random_state=42
    )
    valid_data, test_data = train_test_split(
        valid_data, test_size=0.5, random_state=42
    )

    logging.info(f'Train size: {len(train_data)}')
    logging.info(f'Validation size: {len(valid_data)}')
    logging.info(f'Test size: {len(test_data)}')

    logging.info('Creating model')
    model = get_model(
        config=config, 
        dataset=dataset, 
        embedding_dict=embedding_dict
    )

    logging.info('Dataset stats')
    dataset.print_stats()

    def collate(batch):
        inputs = [item[0] for item in batch]
        labels = torch.LongTensor([item[1] for item in batch])
        text = [item[2] for item in batch]
        return inputs, labels, text

    train_dataloader = DataLoader(
        train_data, shuffle=True, batch_size=config.batch_size, collate_fn=collate)

    eval_dataloader = DataLoader(
        valid_data, batch_size=config.batch_size, collate_fn=collate)
    
    test_dataloader = DataLoader(
        test_data, batch_size=config.batch_size, collate_fn=collate)

    metrics = get_metrics(config.metrics)
    
    trainer = Trainer(
        model=model,
        train_dataset=train_dataloader,
        eval_dataset=eval_dataloader,
        test_dataset=test_dataloader,
        config=config,
        metrics=metrics,
        use_wandb=args.use_wandb
    )

    logging.info('Training model')
    trainer.train()

    logging.info('Evaluating model')
    trainer.evaluate(model=model)


