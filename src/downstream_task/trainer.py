from .utils import get_model, get_metrics
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW, Adamax, Adagrad, Adadelta, SparseAdam, Adam, RMSprop, SGD
from torch.utils.data import TensorDataset, RandomSampler
from transformers import get_linear_schedule_with_warmup
import wandb
from tqdm import tqdm
import os
import logging

logging.basicConfig(level=logging.INFO)


class Trainer:
    def __init__(
            self,
            train_dataset,
            eval_dataset,
            test_dataset,
            dataset,
            embedding_dict,
            config=None,
            sweep_config=None,
            use_wandb=False,
            embedding_path=None,
            dataset_name=None,
            seed=None
    ):
        self.use_wandb = use_wandb
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.dataset = dataset
        self.config = config
        self.sweep_config = sweep_config
        self.embedding_dict = embedding_dict
        self.embedding_path = embedding_path
        self.dataset_name = dataset_name
        self.seed = seed
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def reset_metrics(self, metrics):
        training_metrics = {}
        valid_metrics = {}
        for metric in metrics:
            training_metrics[f"train_{metric.name}"] = 0.0
            valid_metrics[f"valid_{metric.name}"] = 0.0

        return training_metrics, valid_metrics

    def train_epoch(
        self,
        config,
        model,
        optimizer,
        scheduler,
        loss_fn,
        epoch,
        metrics,
        train_dataloader,
        eval_dataloader
    ):
        model.train()
        total_loss = 0
        training_metrics, _ = self.reset_metrics(metrics)

        for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{config.epochs}'):
            inputs, labels, texts = batch
            labels = labels.to(self.device)
            try:
                _, logits = model(inputs, return_activations=True)
            except:
                print(texts)
                print(inputs)
                raise

            loss = loss_fn(logits, labels)

            for metric in metrics:
                training_metrics[f"train_{metric.name}"] += metric.compute(
                    torch.argmax(logits, dim=1).detach(
                    ).cpu(), labels.detach().cpu()
                ).detach().cpu().numpy()

            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            total_loss += loss.item()
        for metric in metrics:
            training_metrics[f"train_{metric.name}"] /= len(train_dataloader)

        logging.info(f'Training loss: {total_loss / len(train_dataloader)}')
        logging.info(f'Training metrics: {training_metrics}')

        if self.use_wandb:
            wandb.log(
                {**training_metrics, 'train_loss': total_loss / len(train_dataloader)})

        self.validate(config, model, loss_fn, metrics, eval_dataloader)

    def create_model_dir(self, config):
        os.makedirs('./models', exist_ok=True)
        embedding_name = config.embedding_path.split('/')[-1].split('.')[0]
        os.makedirs(f'./models/{embedding_name}', exist_ok=True)
        os.makedirs(
            f'./models/{embedding_name}/{config.dataset}', exist_ok=True)

    def collate(self, batch):
        inputs = [item[0] for item in batch]
        labels = torch.LongTensor([item[1] for item in batch])
        text = [item[2] for item in batch]
        return inputs, labels, text

    def train(self):
        if self.use_wandb:
            wandb.init(project='nlp-project')
            if self.config is None:
                config = wandb.config
            else:
                config = self.config
        else:
            config = self.config

        logging.info(f'Config: {config}')
        self.create_model_dir(config)

        train_dataloader = DataLoader(
            self.train_dataset, shuffle=True, batch_size=config.batch_size, collate_fn=self.collate)

        eval_dataloader = DataLoader(
            self.eval_dataset, batch_size=config.batch_size, collate_fn=self.collate)

        logging.info('Creating model')
        model = get_model(
            config=config,
            dataset=self.dataset,
            embedding_dict=self.embedding_dict
        )
        if self.use_wandb:
            wandb.watch(model)

        metrics = get_metrics(config.metrics)

        optimizer = self.get_optimizer(config, model)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=1, eta_min=0, last_epoch=-1
        )
        loss_fn = self.get_loss(config)
        model.to(self.device)

        for epoch in range(config.epochs):
            self.train_epoch(
                config=config,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_fn=loss_fn,
                epoch=epoch,
                metrics=metrics,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader
            )

    def train_with_sweep(self):

        sweep_id = None
        if self.use_wandb:
            sweep_id = wandb.sweep(self.sweep_config, project='nlp-project3')
            # save sweep id into file
            with open('./sweep_id.txt', 'w') as f:
                f.write(sweep_id)
            embed_path = self.embedding_path.split("/")[-1].split(".")[0]
            mkdir_path = f'./models/{embed_path}/{self.dataset_name}/{sweep_id}'
            os.makedirs(mkdir_path, exist_ok=True)

        wandb.agent(sweep_id, function=self.train)

    def train_bert(self):
        if self.use_wandb:
            wandb.init(project='nlp-project')
            if self.config is None:
                config = wandb.config
            else:
                config = self.config
        else:
            config = self.config

        logging.info(f'Config: {config}')
        self.create_model_dir(config)

        logging.info('Creating model')
        model = get_model(
            config=config,
            dataset=self.dataset,
            embedding_dict=self.embedding_dict
        )

        train_input_ids, train_attention_masks, train_labels = model.preprocess_data(
            self.train_dataset)
        valid_input_ids, valid_attention_masks, valid_labels = model.preprocess_data(
            self.eval_dataset)

        train_dataset = TensorDataset(
            train_input_ids, train_attention_masks, train_labels)
        valid_dataset = TensorDataset(
            valid_input_ids, valid_attention_masks, valid_labels)

        train_dataloader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=config.batch_size
        )
        valid_dataloader = DataLoader(
            valid_dataset,
            sampler=RandomSampler(valid_dataset),
            batch_size=config.batch_size
        )

        model = model.get_model()
        if self.use_wandb:
            wandb.watch(model)

        metrics = get_metrics(config.metrics)
        optimizer = self.get_optimizer(config, model)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_dataloader) * config.epochs
        )

        model.to(self.device)

        for epoch in range(config.epochs):
            self.train_epoch_bert(
                config=config,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=metrics,
                train_dataloader=train_dataloader,
                eval_dataloader=valid_dataloader
            )

    def train_epoch_bert(
        self,
        config,
        model,
        optimizer,
        scheduler,
        epoch,
        metrics,
        train_dataloader,
        eval_dataloader
    ):
        model.train()
        total_loss = 0
        training_metrics, _ = self.reset_metrics(metrics)

        for batch in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{config.epochs}'):
            input_ids, attention_masks, labels = batch
            input_ids = input_ids.to(self.device)
            attention_masks = attention_masks.to(self.device)
            labels = labels.to(self.device)

            outputs = model(input_ids, attention_masks, labels=labels)

            loss = outputs[0]
            logits = outputs[1]
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            for metric in metrics:
                training_metrics[f"train_{metric.name}"] += metric.compute(
                    torch.argmax(logits, dim=1).detach(
                    ).cpu(), labels.detach().cpu()
                ).detach().cpu().numpy()

            loss.backward()
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            total_loss += loss.item()
        for metric in metrics:
            training_metrics[f"train_{metric.name}"] /= len(train_dataloader)

        logging.info(f'Training loss: {total_loss / len(train_dataloader)}')
        logging.info(f'Training metrics: {training_metrics}')

        if self.use_wandb:
            wandb.log(
                {**training_metrics, 'train_loss': total_loss / len(train_dataloader)})

        self.validate_bert(config, model, metrics, eval_dataloader)

    def validate_bert(self, config, model, metrics, eval_dataloader):
        best_eval_loss = float('inf')
        eval_loss = 0
        _, eval_metrics = self.reset_metrics(metrics)

        model.eval()
        for batch in tqdm(eval_dataloader, desc=f'Validation'):
            with torch.no_grad():
                input_ids, attention_masks, labels = batch
                input_ids = input_ids.to(self.device)
                attention_masks = attention_masks.to(self.device)
                labels = labels.to(self.device)

                ouputs = model(input_ids, attention_masks, labels=labels)
                loss = ouputs[0]
                logits = ouputs[1]
                eval_loss += loss.item()

                for metric in metrics:
                    eval_metrics[f"valid_{metric.name}"] += metric.compute(
                        torch.argmax(logits, dim=1).detach(
                        ).cpu(), labels.detach().cpu()
                    ).detach().cpu().numpy()

        for metric in metrics:
            eval_metrics[f"valid_{metric.name}"] /= len(eval_dataloader)

        logging.info(f'Validation loss: {eval_loss / len(eval_dataloader)}')
        logging.info(f'Validation metrics: {eval_metrics}')

        if self.use_wandb:
            wandb.log(
                {**eval_metrics, 'valid_loss': eval_loss / len(eval_dataloader)})

        eval_loss /= len(eval_dataloader)
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            if os.path.exists('./sweep_id.txt'):
                with open('./sweep_id.txt', 'r') as f:
                    sweep_id = f.read()
                model.save_pretrained(
                    f'./models/{config.model_path.split("/")[-1]}/{config.dataset}/{sweep_id}/{config.model_name}')
            else:
                model.save_pretrained(
                    f'./models/{config.model_path.split("/")[-1]}/{config.dataset}/{config.model_name}')

    def validate(self, config, model, loss_fn, metrics, eval_dataloader):
        best_eval_loss = float('inf')
        eval_loss = 0
        _, eval_metrics = self.reset_metrics(metrics)

        model.eval()
        for batch in tqdm(eval_dataloader, desc=f'Validation'):
            with torch.no_grad():
                inputs, labels, _ = batch
                labels = labels.to(self.device)

                _, logits = model(inputs, return_activations=True)
                loss = loss_fn(logits, labels)
                eval_loss += loss.item()

                for metric in metrics:
                    eval_metrics[f"valid_{metric.name}"] += metric.compute(
                        torch.argmax(logits, dim=1).detach(
                        ).cpu(), labels.detach().cpu()
                    ).detach().cpu().numpy()

        for metric in metrics:
            eval_metrics[f"valid_{metric.name}"] /= len(eval_dataloader)

        logging.info(f'Validation loss: {eval_loss / len(eval_dataloader)}')
        logging.info(f'Validation metrics: {eval_metrics}')

        if self.use_wandb:
            wandb.log(
                {**eval_metrics, 'valid_loss': eval_loss / len(eval_dataloader)})

        eval_loss /= len(eval_dataloader)
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            if os.path.exists('./sweep_id.txt'):
                with open('./sweep_id.txt', 'r') as f:
                    sweep_id = f.read()
                embed_path = config.embedding_path.split("/")[-1].split(".")[0]
                model_path = f'./models/{embed_path}/{config.dataset}/{sweep_id}/{config.model_name}.pt'  # nopep8
            else:
                model_path = f'./models/{config.embedding_path.split("/")[-1].split(".")[0]}/{config.dataset}/{config.model_name}.pt'  # nopep8

            torch.save(model.state_dict(), model_path)
            # if self.use_wandb:
            #     # save Artifact as weights
            #     artifact = wandb.Artifact(name=model_path, type='weights')
            #     artifact.add_file(local_path=model_path)
            #     wandb.run.log_artifact(artifact)

    def evaluate(self, model, metrics):
        eval_loss = 0
        # load the best model
        model.load_state_dict(
            torch.load(
                f'./models/{self.config.embedding_path.split("/")[-1].split(".")[0]}/{self.config.dataset}/{self.config.model_name}.pt'  # nopep8
            ))
        loss_fn = self.get_loss(self.config)

        test_dataloader = DataLoader(
            self.test_dataset, batch_size=self.config.batch_size, collate_fn=self.collate)

        test_metrics = {}
        for metric in metrics:
            test_metrics[f"test_{metric.name}"] = 0.0

        for batch in tqdm(test_dataloader, desc=f'Evaluation'):
            with torch.no_grad():
                inputs, labels, _ = batch
                labels = labels.to(self.device)

                _, logits = model(inputs, return_activations=True)
                loss = loss_fn(logits, labels)
                eval_loss += loss.item()

                for metric in metrics:
                    test_metrics[f"test_{metric.name}"] += metric.compute(
                        torch.argmax(logits, dim=1).detach(
                        ).cpu(), labels.detach().cpu()
                    ).detach().cpu().numpy()

        eval_loss /= len(test_dataloader)
        for metric in metrics:
            test_metrics[f"test_{metric.name}"] /= len(test_dataloader)

        logging.info(f'Evaluation loss: {eval_loss}')
        logging.info(f'Evaluation metrics: {test_metrics}')
        if self.use_wandb:
            wandb.log({**test_metrics, 'test_loss': eval_loss})
            wandb.finish()

    def evaluate_bert(self, config):
        eval_loss = 0
        # load the best model
        model = get_model(
            config=config,
            dataset=self.dataset,
            embedding_dict=self.embedding_dict
        )
        metrics = get_metrics(config.metrics)

        test_input_ids, test_attention_masks, test_labels = model.preprocess_data(
            self.test_dataset)

        test_dataset = TensorDataset(
            test_input_ids, test_attention_masks, test_labels)

        test_dataloader = DataLoader(
            test_dataset,
            sampler=RandomSampler(test_dataset),
            batch_size=config.batch_size
        )

        model = model.get_model()

        # load the best model
        print(f'./models/{config.model_path.split("/")[-1]}/{config.dataset}/{config.model_name}')  # nopep8
        model.from_pretrained(
            f'./models/{config.model_path.split("/")[-1]}/{config.dataset}/{config.model_name}')  # nopep8

        model.to(self.device)

        test_metrics = {}
        for metric in metrics:
            test_metrics[f"test_{metric.name}"] = 0.0

        for batch in tqdm(test_dataloader, desc=f'Evaluation'):
            with torch.no_grad():
                input_ids, attention_masks, labels = batch
                input_ids = input_ids.to(self.device)
                attention_masks = attention_masks.to(self.device)
                labels = labels.to(self.device)

                ouputs = model(input_ids, attention_masks, labels=labels)
                logits = ouputs[1]

                for metric in metrics:
                    test_metrics[f"test_{metric.name}"] += metric.compute(
                        torch.argmax(logits, dim=1).detach(
                        ).cpu(), labels.detach().cpu()
                    ).detach().cpu().numpy()

        eval_loss /= len(test_dataloader)
        for metric in metrics:
            test_metrics[f"test_{metric.name}"] /= len(test_dataloader)

        logging.info(f'Evaluation loss: {eval_loss}')
        logging.info(f'Evaluation metrics: {test_metrics}')
        # save metrics to file basedon the model path and seed
        with open(f'./models/{config.model_path.split("/")[-1]}/{config.dataset}/{config.model_name}/metrics-{self.seed}.json', 'w') as f:
            import json
            json.dump(test_metrics, f)

    def get_optimizer(self, config, model):
        optimizer = config.optimizer
        optimizer_lr = config.learning_rate

        if optimizer == 'Adam':
            return Adam(model.parameters(), lr=optimizer_lr)
        elif optimizer == 'SGD':
            return SGD(model.parameters(), lr=optimizer_lr)
        elif optimizer == 'RMSprop':
            return RMSprop(model.parameters(), lr=optimizer_lr)
        elif optimizer == 'Adagrad':
            return Adagrad(model.parameters(), lr=optimizer_lr)
        elif optimizer == 'Adadelta':
            return Adadelta(model.parameters(), lr=optimizer_lr)
        elif optimizer == 'Adamax':
            return Adamax(model.parameters(), lr=optimizer_lr)
        elif optimizer == 'SparseAdam':
            return SparseAdam(model.parameters(), lr=optimizer_lr)
        elif optimizer == 'AdamW':
            return AdamW(model.parameters(), lr=optimizer_lr)
        else:
            raise ValueError('Invalid optimizer')

    def get_loss(self, config):
        loss_name = config.loss

        if loss_name == 'CrossEntropyLoss':
            return torch.nn.CrossEntropyLoss()
        if loss_name == 'BinaryCrossEntropyLoss':
            return torch.nn.BCELoss()
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
