# -*- coding:utf-8 -*-
"""
Author: nyLiao
File Created: 2023-10-08
File: efficiency.py
"""
import torch
import torch.nn as nn


class F1Calculator(object):
    r"""F1 score supporting incremental update.

    Args:
        num_classes (int): The number of classes in the classification task.
    """
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.TP = torch.zeros(self.num_classes)
        self.FP = torch.zeros(self.num_classes)
        self.FN = torch.zeros(self.num_classes)

    def update(self, y_true: torch.Tensor, y_pred: torch.Tensor):
        r"""Update the F1 score with new samples.

        Args:
            y_true (Tensor): The true labels.
            y_pred (Tensor): The predicted labels.
        """
        if len(y_true.shape) == 1 or y_true.shape[1] == 1:
            y_true = nn.functional.one_hot(y_true, num_classes=self.num_classes)
        if len(y_pred.shape) == 1 or y_pred.shape[1] == 1:
            y_pred = nn.functional.one_hot(y_pred, num_classes=self.num_classes)
        self.TP += (y_true * y_pred).sum(dim=0).cpu()
        self.FP += ((1 - y_true) * y_pred).sum(dim=0).cpu()
        self.FN += (y_true * (1 - y_pred)).sum(dim=0).cpu()

    def get(self, average: str=None):
        r"""Compute the F1 score over all samples.

        Args:
            average (['micro','macro']): F1 averaging scheme.

        Returns:
            f1 (float): F1 score.
        """
        eps = 1e-10
        if average == 'micro':
            # For multi-class classification, F1 micro is equivalent to accuracy
            f1 = 2 * self.TP.float().sum() / (2 * self.TP.sum() + self.FP.sum() + self.FN.sum() + eps)
            return f1.item()
        elif average == 'macro':
            f1 = 2 * self.TP.float() / (2 * self.TP + self.FP + self.FN + eps)
            return f1.mean().item()
        else:
            raise ValueError('average must be "micro" or "macro"')


# TODO: more metrics [glemos1](https://github.com/facebookresearch/glemos/blob/main/src/performances/node_classification.py), [glemos2](https://github.com/facebookresearch/glemos/blob/main/src/utils/eval_utils.py)
