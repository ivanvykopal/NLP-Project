# import f1, accuracy, precision, recall
from collections import namedtuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, auc
from functools import partial

Metric = namedtuple('Metric', ['name', 'compute'])

def get_metrics(metrics):
    metrics_list = []
    for metric in metrics:
        name = metric['name']
        # delete name from metric
        del metric['name']
        if name == 'accuracy':
            metrics_list.append(
                Metric(name='accuracy', compute=accuracy_score)
            )
        elif name == 'precision':
            metrics_list.append(
                Metric(name='precision', compute=partial(precision_score, **metric))
            )
        elif name == 'recall':
            metrics_list.append(
                Metric(name='recall', compute=partial(recall_score, **metric))
            )
        elif name == 'f1':
            metrics_list.append(
                Metric(name='f1', compute=partial(f1_score, **metric))
            )
        elif name == 'auc':
            metrics_list.append(
                Metric(name='auc', compute=auc)
            )
        elif name == 'roc_auc':
            metrics_list.append(
                Metric(name='roc_auc', compute=partial(roc_auc_score, **metric))
            )
        else:
            raise ValueError('Invalid metric name')
        
    return metrics_list