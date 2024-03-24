# -*- coding:utf-8 -*-
"""
Author: nyLiao
File Created: 2024-02-26
"""
from typing import Tuple
from argparse import Namespace
import logging
import torch.nn as nn
from pyg_spectral.utils import load_import

from .base import TrnBase
from .fullbatch import TrnFullbatchIter
from .minibatch import TrnMinibatchDec
from utils import ResLogger


class ModelLoader(object):
    r"""Loader for nn.Module object.

    Args:
        args.model (str): Model architecture name.
        args.conv (str): Convolution layer name.
    """
    def __init__(self, args: Namespace, res_logger: ResLogger = None) -> None:
        r"""Assigning model identity.
        """
        self.model = args.model
        self.conv = args.conv
        self.logger = logging.getLogger('log')
        self.res_logger = res_logger or ResLogger()

    def get(self, args: Namespace) -> Tuple[nn.Module, TrnBase]:
        r"""Load model with specified arguments.

        Args:
            args.num_features (int): Number of input features.
            args.num_classes (int): Number of output classes.
            args.hidden (int): Number of hidden units.
            args.layer (int): Number of layers.
            args.dp (float): Dropout rate.
        """
        self.logger.debug('-'*20 + f" Loading model: {self} " + '-'*20)

        kwargs = dict(
            conv=self.conv,
            in_channels=args.num_features,
            out_channels=args.num_classes,
            hidden_channels=args.hidden,
            num_layers=args.layer,
            dropout=args.dp,
        )

        if self.model in ['GCN']:
            self.conv = 'GCNConv'   # Sometimes need to manually fix repr for logging
            module_name = 'torch_geometric.nn.models'
            class_name = self.model

            kwargs.pop('conv')
            kwargs.update(dict(
                cached=False,
                improved=False,
                add_self_loops=False,
                normalize=False,))
            trn = TrnFullbatchIter
        # Default to load from `pyg_spectral`
        else:
            module_name = 'pyg_spectral.nn.models'
            class_name = self.model

            if self.model in ['IterGNN']:
                if self.conv in ['FixLinSumAdj', 'VarLinSumAdj']:
                    self.conv = '-'.join((self.conv, args.theta))
                    kwargs.update(dict(
                        theta=(args.theta, args.alpha),
                        K=args.K,
                        dropedge=args.dpe,))
                trn = TrnFullbatchIter
            elif self.model in ['PostMLP']:
                if self.conv in ['FixSumAdj', 'VarSumAdj']:
                    self.conv = '-'.join((self.conv, args.theta))
                    kwargs.update(dict(
                        theta=(args.theta, args.alpha),
                        K=args.K,
                        dropedge=args.dpe,))
                elif self.conv in ['ChebBase']:
                    kwargs.update(dict(
                        K=args.K,
                        alpha=args.alpha,))
                trn = TrnFullbatchIter
            elif self.model in ['PreDecMLP']:
                if self.conv in ['FixSumAdj']:
                    self.conv = '-'.join((self.conv, args.theta))
                    kwargs.update(dict(
                        theta=(args.theta, args.alpha),
                        K=args.K,
                        dropedge=args.dpe,))
                trn = TrnMinibatchDec
            else:
                raise ValueError(f"Model '{self}' not found.")

        model = load_import(class_name, module_name)(**kwargs)

        self.logger.log(logging.LTRN, f"[model]: {model}")
        self.logger.info(f"[trainer]: {trn.__name__}")
        self.res_logger.concat([('model', str(self)),])
        return model, trn

    def __call__(self, *args, **kwargs):
        return self.get(*args, **kwargs)

    def __str__(self) -> str:
        return f"{self.model}:{self.conv}"
