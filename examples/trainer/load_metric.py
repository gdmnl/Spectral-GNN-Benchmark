# -*- coding:utf-8 -*-
"""
Author: nyLiao
File Created: 2024-03-03
"""
from argparse import Namespace
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassF1Score, MultilabelF1Score)


def metric_loader(args: Namespace) -> MetricCollection:
    if args.multi:
        metric = MetricCollection({
            'f1micro': MultilabelF1Score(num_labels=args.num_classes, average='micro'),
            'f1macro': MultilabelF1Score(num_labels=args.num_classes, average='macro'),
        })
    else:
        metric = MetricCollection({
            'f1micro': MulticlassF1Score(num_classes=args.num_classes, average='micro'),
            'f1macro': MulticlassF1Score(num_classes=args.num_classes, average='macro'),
        })
    return metric
