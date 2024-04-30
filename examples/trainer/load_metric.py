# -*- coding:utf-8 -*-
"""
Author: nyLiao
File Created: 2024-03-03
"""
from typing import Tuple, List, Callable, Any
from argparse import Namespace
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassF1Score, MultilabelF1Score, MulticlassAccuracy)



class ResCollection(MetricCollection):
    def compute(self) -> List[Tuple[str, Any, Callable]]:
        r"""Wrap compute output to ResLogger style."""
        dct = self._compute_and_reduce("compute")
        return [(k, v.cpu().numpy(), (lambda x: format(x*100, '.3f'))) for k, v in dct.items()]


def metric_loader(args: Namespace) -> MetricCollection:
    r"""Loader for torchmetrics.Metric object.

    Args:
        args.multi (bool): True for multi-label classification.
        args.num_classes (int): Number of output classes/labels.
    """
    # TODO: more metrics [glemos1](https://github.com/facebookresearch/glemos/blob/main/src/performances/node_classification.py), [glemos2](https://github.com/facebookresearch/glemos/blob/main/src/utils/eval_utils.py)
    if args.multi:
        metric = ResCollection({
            'accuracy': MulticlassAccuracy(num_classes=args.num_classes),
            'f1micro': MultilabelF1Score(num_labels=args.num_classes, average='micro'),
            'f1macro': MultilabelF1Score(num_labels=args.num_classes, average='macro'),
        })
    else:
        metric = ResCollection({
            'accuracy': MulticlassAccuracy(num_classes=args.num_classes),
            'f1micro': MulticlassF1Score(num_classes=args.num_classes, average='micro'),
            'f1macro': MulticlassF1Score(num_classes=args.num_classes, average='macro'),
        })
    return metric
