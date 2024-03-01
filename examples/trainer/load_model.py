# -*- coding:utf-8 -*-
"""
Author: nyLiao
File Created: 2024-02-26
File: load_model.py
"""
from argparse import Namespace
import logging
import torch.nn as nn
from pyg_spectral.utils import load_import

from utils import ResLogger


LTRN = 25


class ModelLoader(object):
    def __init__(self, args: Namespace, res_logger: ResLogger = None) -> None:
        r"""Assigning model identity.

        Args:
            args.model (str): Model architecture name.
            args.conv (str): Convolution layer name.
        """
        self.model = args.model
        self.conv = args.conv
        self.logger = logging.getLogger('log')
        self.res_logger = res_logger or ResLogger()

    def get(self, args: Namespace) -> nn.Module:
        r"""Load model with specified arguments.

        Args:
            args.num_features (int): Number of input features.
            args.num_classes (int): Number of output classes.
            args.hidden (int): Number of hidden units.
            args.layer (int): Number of layers.
            args.dp (float): Dropout rate.
        """
        self.logger.debug('-'*20 + f" Loading model: {self} " + '-'*20)

        # TODO: whether to add batch norm
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
        # Default to load from `pyg_spectral`
        else:
            module_name = 'pyg_spectral.nn.models'
            class_name = self.model

            if self.model in ['IterConv']:
                if self.conv in ['FixLinSumAdj', 'VarLinSumAdj']:
                    self.conv = '-'.join((self.conv, args.theta))
                    kwargs.update(dict(
                        theta=(args.theta, args.alpha),
                        K=args.K,))
            elif self.model in ['DecPostMLP']:
                if self.conv in ['FixSumAdj', 'VarSumAdj']:
                    self.conv = '-'.join((self.conv, args.theta))
                    kwargs.update(dict(
                        theta=(args.theta, args.alpha),
                        K=args.K,))
            else:
                raise ValueError(f"Model '{self}' not found.")

        model = load_import(class_name, module_name)(**kwargs)

        self.logger.log(LTRN, f"[model]: {model}")
        self.res_logger.concat([('model', str(self)),])
        return model

    def __call__(self, *args, **kwargs):
        return self.get(*args, **kwargs)

    def __str__(self) -> str:
        return f"{self.model}:{self.conv}"
