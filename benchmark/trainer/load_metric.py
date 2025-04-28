# -*- coding:utf-8 -*-
"""
Author: nyLiao
File Created: 2024-03-03
"""
import re
from typing import Callable, Any
from argparse import Namespace
import numpy as np
import torch

from ogb.linkproppred import Evaluator as LEvaluator
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


class OGBLEvaluatorNp(LEvaluator):
    def __call__(self, output, label) -> Any:
        output, label = output.cpu().numpy(), label.cpu().numpy()
        if not hasattr(self, 'output'):
            self.output = np.empty((0), dtype=output.dtype)
            self.label = np.empty((0), dtype=label.dtype)
        self.output = np.concatenate((self.output, output), axis=0)
        self.label = np.concatenate((self.label, label), axis=0)

    def to(self, *args: Any, **kwargs: Any) -> 'OGBLEvaluatorNp':
        r"""Override to() to avoid copying the evaluator."""
        return self

    def reset(self, *args: Any, **kwargs: Any) -> 'OGBLEvaluatorNp':
        r"""Override to() to avoid copying the evaluator."""
        self.output = np.empty((0), dtype=np.float32)
        self.label = np.empty((0), dtype=int)
        return self

    def compute(self, *args: Any, **kwargs: Any) -> 'OGBLEvaluatorNp':
        r"""Override to() to avoid copying the evaluator."""
        input_dict = {
            'y_pred_pos': self.output[self.label == 1],
            'y_pred_neg': self.output[self.label == 0],
        }
        if self.eval_metric == 'mrr':
            dim1 = int(len(input_dict['y_pred_neg']) / len(input_dict['y_pred_pos']))
            # input_dict['y_pred_pos'] = np.lib.stride_tricks.as_strided(
            #     input_dict['y_pred_pos'],
            #     shape=(input_dict['y_pred_pos'].size, dim1),
            #     strides=(input_dict['y_pred_pos'].itemsize, 0))
            input_dict['y_pred_neg'] = input_dict['y_pred_neg'].reshape(-1, dim1)
        dct = super().eval(input_dict)

        res_lst = []
        for k, v in dct.items():
            k = re.sub(r'_list$', '', k)
            if isinstance(v, list) or isinstance(v, np.ndarray):
                res_lst.append((f"s_{k}{self.postfix}", np.mean(v), (lambda x: format(x*100, '.3f'))))
            else:
                res_lst.append((f"s_{k}{self.postfix}", v, (lambda x: format(x*100, '.3f'))))
        self.data = res_lst
        return res_lst

    def clone(self, postfix: str) -> 'OGBLEvaluatorNp':
        import copy
        ins = copy.copy(self)
        ins.postfix = postfix
        return ins


class OGBLEvaluator(OGBLEvaluatorNp):
    # torch + on GPU
    def __call__(self, output, label) -> Any:
        output, label = output.detach(), label.detach()
        if not hasattr(self, 'output'):
            self.output = torch.empty((0,), dtype=output.dtype, device=output.device)
            self.label = torch.empty((0,), dtype=label.dtype, device=label.device)
        self.output = torch.cat((self.output, output), dim=0)
        self.label = torch.cat((self.label, label), dim=0)

    def to(self, *args: Any, **kwargs: Any) -> 'OGBLEvaluator':
        r"""Move the evaluator to the specified device."""
        self.device = args[0]
        if hasattr(self, 'output'):
            self.output = self.output.to(self.device)
            self.label = self.label.to(self.device)
        return self

    def reset(self, *args: Any, **kwargs: Any) -> 'OGBLEvaluator':
        r"""Reset the evaluator."""
        device = self.output.device if hasattr(self, 'output') else self.device
        self.output = torch.empty((0,), dtype=torch.float32, device=device)
        self.label = torch.empty((0,), dtype=torch.int, device=device)
        return self

    def compute(self, *args: Any, **kwargs: Any) -> 'OGBLEvaluator':
        r"""Override to() to avoid copying the evaluator."""
        input_dict = {
            'y_pred_pos': self.output[self.label == 1],
            'y_pred_neg': self.output[self.label == 0],
        }
        if self.eval_metric == 'mrr':
            dim1 = int(len(input_dict['y_pred_neg']) / len(input_dict['y_pred_pos']))
            input_dict['y_pred_neg'] = input_dict['y_pred_neg'].view(-1, dim1)
        dct = super().eval(input_dict)

        res_lst = []
        for k, v in dct.items():
            k = re.sub(r'_list$', '', k)
            if isinstance(v, list):
                v = torch.tensor(v).mean().item()
            elif isinstance(v, torch.Tensor):
                v = v.mean().item()
            res_lst.append((f"s_{k}{self.postfix}", v, (lambda x: format(x*100, '.3f'))))
        self.data = res_lst
        return res_lst


def metric_loader(args: Namespace) -> MetricCollection:
    r"""Loader for :class:`torchmetrics.Metric` object.

    Args:
        args: Configuration arguments.

            * args.multi (bool): True for multi-label classification.
            * args.out_channels (int): Number of output classes/labels.
    """
    if args.data.startswith('ogbl-'):
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
