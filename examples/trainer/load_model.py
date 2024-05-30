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
from .fullbatch import TrnFullbatch
from .minibatch import TrnMinibatch
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
        self.conv_repr = args.conv_repr

        self.logger = logging.getLogger('log')
        self.res_logger = res_logger or ResLogger()

    def _resolve_import(self, args: Namespace) -> Tuple[str, str, dict, TrnBase]:
        # >>>>>>>>>>
        if self.model in ['GCN', 'MLP']:
            conv_dct = {
                'GCN': 'GCNConv',
                'MLP': 'Identity',
            }
            self.conv_repr = conv_dct[self.model]   # Sometimes need to manually fix repr for logging
            module_name = 'torch_geometric.nn.models'
            class_name = self.model
            kwargs = dict(
                in_channels=args.num_features,
                out_channels=args.num_classes,
                hidden_channels=args.hidden,
                num_layers=args.in_layers+args.out_layers,
                dropout=args.dp_lin,
            )
            trn = TrnFullbatch

        # Default to load from `pyg_spectral`
        else:
            module_name = 'pyg_spectral.nn.models'
            class_name = self.model
            kwargs = dict(
                conv=self.conv,
                num_hops=args.num_hops,
                in_layers=args.in_layers,
                out_layers=args.out_layers,
                in_channels=args.num_features,
                out_channels=args.num_classes,
                hidden_channels=args.hidden,
                dropout_lin=args.dp_lin,
                dropout_conv=args.dp_conv,
            )

            # Parse conv args
            if self.conv in ['AdjConv', 'ChebConv', 'HornerConv', 'ClenshawConv']:
                kwargs.update(dict(
                    alpha=args.alpha,))
            elif self.conv in ['JacobiConv', 'AdjDiffConv', 'AdjiConv', 'Adji2Conv', 'AdjResConv']:
                kwargs.update(dict(
                    alpha=args.alpha,
                    beta=args.beta,))
                if self.conv == 'Adji2Conv':
                    kwargs['num_hops'] = int(kwargs['num_hops'] / 2)

            # Parse model args
            if self.model in ['Iterative', 'ACMGNN']:
                trn = TrnFullbatch
            elif self.model in ['DecoupledFixed', 'DecoupledVar', 'AdaGNN']:
                kwargs.update(dict(
                    theta_scheme=args.theta_scheme,
                    theta_param=args.theta_param,))
                trn = TrnFullbatch
            elif self.model in ['DecoupledFixedCompose', 'DecoupledVarCompose']:
                kwargs.update(dict(
                    theta_scheme=args.theta_scheme,
                    theta_param=args.theta_param,
                    combine=args.combine,))
                trn = TrnFullbatch
            elif self.model in ['PrecomputedFixed', 'PrecomputedVar']:
                kwargs.update(dict(
                    theta_scheme=args.theta_scheme,
                    theta_param=args.theta_param,))
                trn = TrnMinibatch
            else:
                raise ValueError(f"Model '{self}' not found.")
        # <<<<<<<<<<
        return class_name, module_name, kwargs, trn

    def get(self, args: Namespace) -> Tuple[nn.Module, TrnBase]:
        r"""Load model with specified arguments.

        Args:
            args.num_hops (int): Number of conv hops.
            args.in_layers (int): Number of MLP layers before conv.
            args.out_layers (int): Number of MLP layers after conv.
            args.num_features (int): Number of input features.
            args.num_classes (int): Number of output classes.
            args.hidden (int): Number of hidden units.
            args.dp_[lin/conv] (float): Dropout rate for linear/conv.
        """
        self.logger.debug('-'*20 + f" Loading model: {self} " + '-'*20)

        class_name, module_name, kwargs, trn = self._resolve_import(args)
        model = load_import(class_name, module_name)(**kwargs)
        if hasattr(model, 'reset_parameters'):
            model.reset_parameters()
        if hasattr(model, 'reset_cache'):
            model.reset_cache()

        self.logger.log(logging.LTRN, f"[model]: {str(self)}")
        self.logger.log(logging.LTRN, str(model))
        self.logger.info(f"[trainer]: {trn.__name__}")
        self.res_logger.concat([('model', self.model), ('conv', self.conv_repr)])
        return model, trn

    def __call__(self, *args, **kwargs):
        return self.get(*args, **kwargs)

    def __str__(self) -> str:
        return f"{self.model}:{self.conv_repr}"


class ModelLoader_Trial(ModelLoader):
    r"""Reuse necessary data for multiple runs.
    """
    def get(self, args: Namespace) -> Tuple[nn.Module, TrnBase]:
        self.signature_lst = ['num_hops', 'in_layers', 'out_layers', 'hidden', 'dp_lin', 'dp_conv']
        self.signature = {key: args.__dict__[key] for key in self.signature_lst}

        class_name, module_name, kwargs, trn = self._resolve_import(args)
        model = load_import(class_name, module_name)(**kwargs)
        model.reset_parameters()
        model.reset_cache()

        self.res_logger.concat([('model', self.model), ('conv', self.conv_repr)])
        return model, trn

    def update(self, args: Namespace, model: nn.Module) -> nn.Module:
        signature = {key: args.__dict__[key] for key in self.signature_lst}
        if self.signature != signature:
            self.signature = signature
            model, _ = self.get(args)
            return model

        if hasattr(model, 'reset_parameters'):
            model.reset_parameters()
        return model
