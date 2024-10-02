# -*- coding:utf-8 -*-
"""
Author: nyLiao
File Created: 2024-03-03
"""
from typing import Tuple, List, Callable, Any
from argparse import Namespace
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy, MultilabelAccuracy, BinaryAccuracy,
    MulticlassF1Score, MultilabelF1Score, BinaryF1Score,
    MulticlassAUROC, MultilabelAUROC, BinaryAUROC,
    MulticlassAveragePrecision, MultilabelAveragePrecision, BinaryAveragePrecision,
)


class ResCollection(MetricCollection):
    def compute(self) -> List[Tuple[str, Any, Callable]]:
        r"""Wrap compute output to :class:`ResLogger` style."""
        dct = self._compute_and_reduce("compute")
        return [(k, v.cpu().numpy(), (lambda x: format(x*100, '.3f'))) for k, v in dct.items()]


def metric_loader(args: Namespace) -> MetricCollection:
    r"""Loader for :class:`torchmetrics.Metric` object.

    Args:
        args: Configuration arguments.

            args.multi (bool): True for multi-label classification.
            args.num_classes (int): Number of output classes/labels.
    """
    # FEATURE: more metrics [glemos1](https://github.com/facebookresearch/glemos/blob/main/src/performances/node_classification.py), [glemos2](https://github.com/facebookresearch/glemos/blob/main/src/utils/eval_utils.py)
    if args.multi:
        metric = ResCollection({
            's_acc': MultilabelAccuracy(num_classes=args.num_classes),
            's_f1i': MultilabelF1Score(num_labels=args.num_classes, average='micro'),
            # 's_f1a': MultilabelF1Score(num_labels=args.num_classes, average='macro'),
            's_auroc': MultilabelAUROC(num_classes=args.num_classes),
            's_ap': MultilabelAveragePrecision(num_classes=args.num_classes),
        })
    elif args.num_classes == 1:
        metric = ResCollection({
            's_acc': BinaryAccuracy(),
            's_f1i': BinaryF1Score(),
            's_auroc': BinaryAUROC(),
            's_ap': BinaryAveragePrecision(),
        })
    else:
        metric = ResCollection({
            's_acc': MulticlassAccuracy(num_classes=args.num_classes),
            's_f1i': MulticlassF1Score(num_classes=args.num_classes, average='micro'),
            # 's_f1a': MulticlassF1Score(num_classes=args.num_classes, average='macro'),
            's_auroc': MulticlassAUROC(num_classes=args.num_classes),
            's_ap': MulticlassAveragePrecision(num_classes=args.num_classes),
        })
    return metric
