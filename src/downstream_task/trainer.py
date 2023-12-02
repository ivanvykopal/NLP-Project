import torch
from torch.optim import AdamW, Adamax, Adagrad, Adadelta, SparseAdam, Adam, RMSprop, SGD
import wandb
from tqdm import tqdm
import os
import logging

logging.basicConfig(level=logging.INFO)

class Trainer:
    def __init__(
            self, 
            model, 
            train_dataset, 
            eval_dataset,
            test_dataset, 
            config, 
            metrics, 
            use_wandb=False
            ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.use_wandb = use_wandb
        self.config = config
        self.metrics = metrics
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def reset_metrics(self, metrics):
        training_metrics = {}
        valid_metrics = {}
        for metric in metrics:
              training_metrics[f"train_{metric.name}"] = 0.0
              valid_metrics[f"valid_{metric.name}"] = 0.0

        return training_metrics, valid_metrics

    def train_epoch(self, optimizer, scheduler, loss_fn, epoch):
            self.model.train()
            total_loss = 0
            training_metrics, _ = self.reset_metrics(self.metrics)

            for batch in tqdm(self.train_dataset, desc=f'Epoch {epoch + 1}/{self.config.epochs}'):
                inputs, labels, _ = batch
                labels = labels.to(self.device)
                
                _, logits = self.model(inputs, return_activations=True)
                loss = loss_fn(logits, labels)

                for metric in self.metrics:
                    training_metrics[f"train_{metric.name}"] += metric.compute(
                        torch.argmax(logits, dim=1).detach().cpu(), labels.detach().cpu()
                    ).detach().cpu().numpy()

                loss.backward()
                optimizer.step()
                scheduler.step()
                self.model.zero_grad()

                total_loss += loss.item()
            for metric in self.metrics:
                training_metrics[f"train_{metric.name}"] /= len(self.train_dataset)

            logging.info(f'Training loss: {total_loss / len(self.train_dataset)}')
            logging.info(f'Training metrics: {training_metrics}')

            if self.use_wandb:
                wandb.log({**training_metrics, 'train_loss': total_loss / len(self.train_dataset)})

            self.validate(loss_fn)

    def train(self):
        os.makedirs('./models', exist_ok=True)
        embedding_name = self.config.embedding_path.split('/')[-1].split('.')[0]
        os.makedirs(f'./models/{embedding_name}', exist_ok=True)
        os.makedirs(f'./models/{embedding_name}/{self.config.dataset}', exist_ok=True)

        if self.use_wandb:
            wandb.init(project='nlp-project')
            wandb.watch(self.model)

        optimizer = self.get_optimizer()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=1, eta_min=0, last_epoch=-1
        )
        loss_fn = self.get_loss()
        self.model.to(self.device)

        for epoch in range(self.config.epochs):
            self.train_epoch(optimizer, scheduler,  loss_fn, epoch)
    
    def validate(self, loss_fn):
        best_eval_loss = float('inf')
        eval_loss = 0
        _, eval_metrics = self.reset_metrics(self.metrics)

        self.model.eval()
        for batch in tqdm(self.eval_dataset, desc=f'Validation'):
            with torch.no_grad():
                inputs, labels, _ = batch
                labels = labels.to(self.device)

                _, logits = self.model(inputs, return_activations=True)
                loss = loss_fn(logits, labels)
                eval_loss += loss.item()

                for metric in self.metrics:
                    eval_metrics[f"valid_{metric.name}"] += metric.compute(
                        torch.argmax(logits, dim=1).detach().cpu(), labels.detach().cpu()
                    ).detach().cpu().numpy()
    
        for metric in self.metrics:
            eval_metrics[f"valid_{metric.name}"] /= len(self.eval_dataset)

        logging.info(f'Validation loss: {eval_loss / len(self.eval_dataset)}')
        logging.info(f'Validation metrics: {eval_metrics}')

        if self.use_wandb:
            wandb.log({**eval_metrics, 'valid_loss': eval_loss / len(self.eval_dataset)})

        eval_loss /= len(self.eval_dataset)
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            model_path = f'./models/{self.config.embedding_path.split("/")[-1].split(".")[0]}/{self.config.dataset}/{self.config.model_name}.pt'
            torch.save(self.model.state_dict(), model_path)
            if self.use_wandb:
                # save Artifact as weights
                artifact = wandb.Artifact(name=model_path, type='weights')
                artifact.add_file(local_path=model_path)
                wandb.run.log_artifact(artifact)
    
    def evaluate(self, model):
        eval_loss = 0
        # load the best model
        model.load_state_dict(
            torch.load(
                f'./models/{self.config.embedding_path.split("/")[-1].split(".")[0]}/{self.config.dataset}/{self.config.model_name}.pt'
            ))
        loss_fn = self.get_loss()
        
        test_metrics = {}
        for metric in self.metrics:
              test_metrics[f"test_{metric.name}"] = 0.0

        for batch in tqdm(self.test_dataset, desc=f'Evaluation'):
            with torch.no_grad():
                inputs, labels, _ = batch
                labels = labels.to(self.device)

                _, logits = model(inputs, return_activations=True)
                loss = loss_fn(logits, labels)
                eval_loss += loss.item()
                
                for metric in self.metrics:
                    test_metrics[f"test_{metric.name}"] += metric.compute(
                        torch.argmax(logits, dim=1).detach().cpu(), labels.detach().cpu()
                    ).detach().cpu().numpy()

        eval_loss /= len(self.eval_dataset)
        for metric in self.metrics:
            test_metrics[f"test_{metric.name}"] /= len(self.eval_dataset)

        logging.info(f'Evaluation loss: {eval_loss}')
        logging.info(f'Evaluation metrics: {test_metrics}')
        if self.use_wandb:
            wandb.log({**test_metrics, 'test_loss': eval_loss})
            wandb.finish()
    
    def get_optimizer(self):
        optimizer = self.config.optimizer
        optimizer_lr = self.config.learning_rate

        if optimizer == 'Adam':
            return Adam(self.model.parameters(), lr=optimizer_lr)
        elif optimizer == 'SGD':
            return SGD(self.model.parameters(), lr=optimizer_lr)
        elif optimizer == 'RMSprop':
            return RMSprop(self.model.parameters(), lr=optimizer_lr)
        elif optimizer == 'Adagrad':
            return Adagrad(self.model.parameters(), lr=optimizer_lr)
        elif optimizer == 'Adadelta':
            return Adadelta(self.model.parameters(), lr=optimizer_lr)
        elif optimizer == 'Adamax':
            return Adamax(self.model.parameters(), lr=optimizer_lr)
        elif optimizer == 'SparseAdam':
            return SparseAdam(self.model.parameters(), lr=optimizer_lr)
        elif optimizer == 'AdamW':
            return AdamW(self.model.parameters(), lr=optimizer_lr)
        else:
            raise ValueError('Invalid optimizer')
        
    def get_loss(self):
        loss_name = self.config.loss

        if loss_name == 'CrossEntropyLoss':
            return torch.nn.CrossEntropyLoss()
        elif loss_name == 'NLLLoss':
            return torch.nn.NLLLoss()
        elif loss_name == 'MSELoss':
            return torch.nn.MSELoss()
        elif loss_name == 'L1Loss':
            return torch.nn.L1Loss()
        elif loss_name == 'SmoothL1Loss':
            return torch.nn.SmoothL1Loss()
        elif loss_name == 'PoissonNLLLoss':
            return torch.nn.PoissonNLLLoss()
        elif loss_name == 'KLDivLoss':
            return torch.nn.KLDivLoss()
        elif loss_name == 'BCELoss':
            return torch.nn.BCELoss()
        elif loss_name == 'BCEWithLogitsLoss':
            return torch.nn.BCEWithLogitsLoss()
        elif loss_name == 'MarginRankingLoss':
            return torch.nn.MarginRankingLoss()
        elif loss_name == 'HingeEmbeddingLoss':
            return torch.nn.HingeEmbeddingLoss()
        elif loss_name == 'MultiLabelMarginLoss':
            return torch.nn.MultiLabelMarginLoss()
        elif loss_name == 'SmoothL1Loss':
            return torch.nn.SmoothL1Loss()
        elif loss_name == 'SoftMarginLoss':
            return torch.nn.SoftMarginLoss()
        elif loss_name == 'MultiLabelSoftMarginLoss':
            return torch.nn.MultiLabelSoftMarginLoss()
        elif loss_name == 'CosineEmbeddingLoss':
            return torch.nn.CosineEmbeddingLoss()
        elif loss_name == 'MultiMarginLoss':
            return torch.nn.MultiMarginLoss()
        elif loss_name == 'TripletMarginLoss':
            return torch.nn.TripletMarginLoss()
        else:
            raise ValueError('Invalid loss')
            
