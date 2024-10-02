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
    r"""Loader for :class:`torch.nn.Module` object.

    Args:
        args: Configuration arguments.

            args.model (str): Model architecture name.
            args.conv (str): Convolution layer name.
        res_logger: Logger for results.
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
        if self.model in ['GCN', 'GraphSAGE', 'GIN', 'GAT', 'PNA', 'MLP']:
            from pyg_spectral.nn.models_pyg import kwargs_default
            # manually fix repr for logging
            conv_dct = {
                'GCN': 'GCNConv',
                'GraphSAGE': 'SAGEConv',
                'GIN': 'GINConv',
                'GAT': 'GATConv',
                'PNA': 'PNAConv',
                'MLP': 'Identity',
            }
            self.conv_repr = '-'.join((conv_dct[self.model], args.theta_scheme))

            # workaround for aligning MLP num_layers
            if args.theta_scheme == 'ones':
                num_layers = args.in_layers + args.out_layers
                trn = TrnFullbatch
            elif args.theta_scheme == 'appr':
                num_layers = args.in_layers + args.num_hops + args.out_layers
                trn = TrnFullbatch
            else:
                num_layers = args.out_layers
                trn = TrnMinibatch

            module_name = 'torch_geometric.nn.models'
            class_name = self.model
            kwargs = dict(
                criterion='BCEWithLogitsLoss' if args.num_classes == 1 else 'CrossEntropyLoss',
                in_channels=args.num_features,
                out_channels=args.num_classes,
                hidden_channels=args.hidden,
                num_layers=num_layers,
                dropout=args.dp_lin,
            )
            if self.model in kwargs_default:
                for k, v in kwargs_default[self.model].items():
                    kwargs.setdefault(k, v)

        elif self.model in ['ChebNet', ]:
            from pyg_spectral.nn.models_pyg import kwvars
            module_name = 'pyg_spectral.nn.models_pyg'
            class_name = self.model
            kwvar = kwvars[self.model]
            kwargs = {k: args.__dict__[v] for k, v in kwvar.items()}
            kwargs['criterion'] = 'BCELoss' if args.num_classes == 1 else 'NLLLoss'
            trn = TrnFullbatch

        # Default to load from `pyg_spectral`
        else:
            module_name = 'pyg_spectral.nn.models'
            class_name = self.model
            kwargs = dict(
                criterion='BCELoss' if args.num_classes == 1 else 'NLLLoss',
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
            for conv in self.conv.split(','):
                if conv in ['AdjConv', 'ChebConv', 'HornerConv', 'ClenshawConv', 'ACMConv']:
                    kwargs.update(dict(
                        alpha=args.alpha,))
                elif conv in ['JacobiConv', 'AdjDiffConv', 'AdjiConv', 'Adji2Conv', \
                            'AdjSkipConv', 'AdjSkip2Conv', 'AdjResConv']:
                    kwargs.update(dict(
                        alpha=args.alpha,
                        beta=args.beta,))
                    if conv == 'Adji2Conv':
                        kwargs['num_hops'] = int(kwargs['num_hops'] / 2)

            # Parse model args
            if self.model in ['Iterative', 'IterativeCompose', 'ACMGNN', 'ACMGNNDec']:
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
            elif self.model in ['PrecomputedFixed', 'PrecomputedVar', 'CppCompFixed']:
                kwargs.update(dict(
                    theta_scheme=args.theta_scheme,
                    theta_param=args.theta_param,))
                trn = TrnMinibatch
            elif self.model in ['PrecomputedFixedCompose', 'PrecomputedVarCompose']:
                kwargs.update(dict(
                    theta_scheme=args.theta_scheme,
                    theta_param=args.theta_param,
                    combine=args.combine,))
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

        Updates:
            args.criterion (str): Criterion for loss calculation
        """
        self.logger.debug('-'*20 + f" Loading model: {self} " + '-'*20)

        class_name, module_name, kwargs, trn = self._resolve_import(args)
        args.criterion = kwargs.pop('criterion')

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
        args.criterion = kwargs.pop('criterion')

        model = load_import(class_name, module_name)(**kwargs)
        if hasattr(model, 'reset_parameters'):
            model.reset_parameters()
        if hasattr(model, 'reset_cache'):
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
