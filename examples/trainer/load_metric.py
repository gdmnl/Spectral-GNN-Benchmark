# -*- coding:utf-8 -*-
"""
Author: nyLiao
File Created: 2024-03-03
"""
from typing import Tuple, List, Callable, Any
from argparse import Namespace
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassF1Score, MultilabelF1Score)



class ResCollection(MetricCollection):
    def compute(self) -> List[Tuple[str, Any, Callable]]:
        r"""Wrap compute output to ResLogger style."""
        dct = self._compute_and_reduce("compute")
        return [(k, v.numpy(), (lambda x: format(x, '.4f'))) for k, v in dct.items()]


def metric_loader(args: Namespace) -> MetricCollection:
    r"""Loader for torchmetrics.Metric object.

    Args:
        args.multi (bool): True for multi-label classification.
        args.num_classes (int): Number of output classes/labels.
    """
    if args.multi:
        metric = ResCollection({
            'f1micro': MultilabelF1Score(num_labels=args.num_classes, average='micro'),
            'f1macro': MultilabelF1Score(num_labels=args.num_classes, average='macro'),
        })
    else:
        metric = ResCollection({
            'f1micro': MulticlassF1Score(num_classes=args.num_classes, average='micro'),
            'f1macro': MulticlassF1Score(num_classes=args.num_classes, average='macro'),
        })
    return metric