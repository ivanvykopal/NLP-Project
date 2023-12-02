import yaml
import configparser as ConfigParser
from collections import namedtuple
from embeddings.utils import load_stopwords
import torchmetrics
from gensim.parsing.preprocessing import remove_stopwords, strip_punctuation, strip_tags, strip_multiple_whitespaces, strip_numeric, strip_short, strip_short, strip_numeric, strip_punctuation, strip_tags, strip_multiple_whitespaces, remove_stopwords

from .XLMRoBERTa import XLMRoBERTa
from .XLMMLM100 import XLMMLM100
from .mBERT import mBERT
from .LSTM import LSTMClassifier


WandbConfig = namedtuple(
    'WandbConfig', ['WANDB_API_KEY', 'WANDB_USERNAME', 'WANDB_DIR'])
Metric = namedtuple('Metric', ['name', 'compute'])


def get_wandb_config(path):
    config = ConfigParser.ConfigParser()
    config.read(path)

    return WandbConfig(
        WANDB_API_KEY=config.get('wandb', 'WANDB_API_KEY'),
        WANDB_USERNAME=config.get('wandb', 'WANDB_USERNAME'),
        WANDB_DIR=config.get('wandb', 'WANDB_DIR')
    )

def get_config(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def get_model(config, dataset, embedding_dict=None):
    model_name = config.model_name
    embedding_dim = config.embedding_dim if hasattr(config, 'embedding_dim') else None
    hidden_dim = config.hidden_dim if hasattr(config, 'hidden_dim') else None
    n_layers = config.n_layers if hasattr(config, 'n_layers') else None
    output_dim = config.output_dim if hasattr(config, 'output_dim') else None
    bidirectional = config.bidirectional if hasattr(config, 'bidirectional') else None

    # unique_words = get_unique_words(list(dataset.data['claim']), dataset.language)
    # # add <pad> token
    # unique_words.append('<pad>')
    # vocabulary = create_vocabulary(unique_words)
    vocabulary = dataset.token2idx

    if model_name.lower() == 'lstmclassifier':
        return LSTMClassifier(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            output_dim=output_dim,
            bidirectional=bidirectional,
            embedding_dict=embedding_dict,
            vocabulary=vocabulary,
            padding_idx=dataset.token2idx['<pad>'],
            batch_size=config.batch_size,
            device=config.device
        )
    elif model_name.lower() == 'mbert':
        return mBERT(config)
    elif model_name.lower() == 'xlmmlm100':
        return XLMMLM100(config)
    elif model_name.lower() == 'xlmroberta':
        return XLMRoBERTa(config)
    else:
        raise ValueError('Invalid model name')
    

def get_metrics(metrics):
    metrics_list = []
    for metric in metrics:
        print(metric)
        name = metric['name']
        # delete name from metric
        del metric['name']
        if name == 'accuracy':
            metrics_list.append(
                Metric(name='accuracy', compute=torchmetrics.Accuracy(**metric))
            )
        elif name == 'precision':
            metrics_list.append(
                Metric(name='precision', compute=torchmetrics.Precision(**metric))
            )
        elif name == 'recall':
            metrics_list.append(
                Metric(name='recall', compute=torchmetrics.Recall(**metric))
            )
        elif name == 'f1':
            metrics_list.append(
                Metric(name='f1', compute=torchmetrics.F1Score(**metric))
            )
        elif name == 'auc':
            metrics_list.append(
                Metric(name='auc', compute=torchmetrics.AUROC())
            )
        elif name == 'prc':
            metrics_list.append(
                Metric(name='prc', compute=torchmetrics.PrecisionRecallCurve())
            )
        elif name == 'confusion_matrix':
            metrics_list.append(
                Metric(name='confusion_matrix', compute=torchmetrics.ConfusionMatrix())
            )
        elif name == 'roc_auc':
            metrics_list.append(
                Metric(name='roc_auc', compute=torchmetrics.ROC())
            )
        else:
            raise ValueError('Invalid metric name')
        
    return metrics_list


# create vocabulary from the list of words
def create_vocabulary(words):
    vocabulary = {}
    for word in words:
        if word not in vocabulary:
            vocabulary[word] = len(vocabulary)
    return vocabulary


def preprocess_string(text, language, source_path='../data/stopwords/'):
    stopwords = load_stopwords(language, source_path)

    data = text.lower()
    data = data.replace('„', '').replace('“', '')
    data = strip_tags(data)
    data = strip_punctuation(data)
    data = strip_multiple_whitespaces(data)
    data = strip_numeric(data)
    data = remove_stopwords(data, stopwords=stopwords)
    data = strip_short(data, minsize=3)

    return data.split()
        

def get_unique_words(data, language):
    return list(
        set([
                word for claim in data
                for word in preprocess_string(claim, language)
            ]))
    

