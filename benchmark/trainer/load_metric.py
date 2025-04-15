# -*- coding:utf-8 -*-
"""
Author: nyLiao
File Created: 2024-03-03
"""
from typing import Callable, Any
from argparse import Namespace
from ogb.linkproppred import Evaluator
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy, MultilabelAccuracy, BinaryAccuracy,
    MulticlassF1Score, MultilabelF1Score, BinaryF1Score,
    MulticlassAUROC, MultilabelAUROC, BinaryAUROC,
    MulticlassAveragePrecision, MultilabelAveragePrecision, BinaryAveragePrecision,
)


class ResCollection(MetricCollection):
    def compute(self) -> list[tuple[str, Any, Callable]]:
        r"""Wrap compute output to :class:`ResLogger` style."""
        dct = self._compute_and_reduce("compute")
        return [(k, v.cpu().numpy(), (lambda x: format(x*100, '.3f'))) for k, v in dct.items()]


class OGBLEvaluator(Evaluator):
    def __call__(self, output, label) -> Any:
        import numpy as np

        output, label = output.cpu().numpy(), label.cpu().numpy()
        input_dict = {
            'y_pred_pos': output[label == 1],
            'y_pred_neg': output[label == 0],
        }
        dct = super().eval(input_dict)
        res_lst = []

        for k, v in dct.items():
            if isinstance(v, list):
                res_lst.append((f"s_{k}{self.postfix}", np.mean(v), (lambda x: format(x*100, '.3f'))))
            else:
                res_lst.append((f"s_{k}{self.postfix}", v, (lambda x: format(x*100, '.3f'))))
        self.data = res_lst
        return res_lst

    def to(self, *args: Any, **kwargs: Any) -> 'OGBLEvaluator':
        r"""Override to() to avoid copying the evaluator."""
        return self

    def reset(self, *args: Any, **kwargs: Any) -> 'OGBLEvaluator':
        r"""Override to() to avoid copying the evaluator."""
        return self

    def compute(self, *args: Any, **kwargs: Any) -> 'OGBLEvaluator':
        r"""Override to() to avoid copying the evaluator."""
        return self.data

    def clone(self, postfix: str) -> 'OGBLEvaluator':
        import copy
        ins = copy.copy(self)
        ins.postfix = postfix
        return ins


def metric_loader(args: Namespace) -> MetricCollection:
    r"""Loader for :class:`torchmetrics.Metric` object.

    Args:
        args: Configuration arguments.

            * args.multi (bool): True for multi-label classification.
            * args.out_channels (int): Number of output classes/labels.
    """
    if args.data in ['ogbl-collab', 'ogbl-ddi', 'ogbl-ppa', 'ogbl-citation2']:
        return OGBLEvaluator(args.data)

    # FEATURE: more metrics [glemos1](https://github.com/facebookresearch/glemos/blob/main/src/performances/node_classification.py), [glemos2](https://github.com/facebookresearch/glemos/blob/main/src/utils/eval_utils.py)
    if args.multi:
        metric = ResCollection({
            's_acc': MultilabelAccuracy(num_classes=args.out_channels),
            's_f1i': MultilabelF1Score(num_labels=args.out_channels, average='micro'),
            # 's_f1a': MultilabelF1Score(num_labels=args.out_channels, average='macro'),
            's_auroc': MultilabelAUROC(num_classes=args.out_channels),
            's_ap': MultilabelAveragePrecision(num_classes=args.out_channels),
        })
    elif args.out_channels == 1:
        metric = ResCollection({
            's_acc': BinaryAccuracy(),
            's_f1i': BinaryF1Score(),
            's_auroc': BinaryAUROC(),
            's_ap': BinaryAveragePrecision(),
        })
    else:
        metric = ResCollection({
            's_acc': MulticlassAccuracy(num_classes=args.out_channels),
            's_f1i': MulticlassF1Score(num_classes=args.out_channels, average='micro'),
            # 's_f1a': MulticlassF1Score(num_classes=args.out_channels, average='macro'),
            's_auroc': MulticlassAUROC(num_classes=args.out_channels),
            's_ap': MulticlassAveragePrecision(num_classes=args.out_channels),
        })
    return metric
