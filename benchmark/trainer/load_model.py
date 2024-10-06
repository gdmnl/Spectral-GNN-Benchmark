# -*- coding:utf-8 -*-
"""
Author: nyLiao
File Created: 2024-02-26
"""
from argparse import Namespace
import logging
import torch.nn as nn
from pyg_spectral.nn import get_model_regi, get_nn_name, set_pargs
from pyg_spectral.utils import load_import

from .base import TrnBase
from .fullbatch import TrnFullbatch
from .minibatch import TrnMinibatch
from utils import ResLogger


class ModelLoader(object):
    r"""Loader for :class:`torch.nn.Module` object.

    Args:
        args: Configuration arguments.

            * args.model (str): Model architecture name.
            * args.conv (str): Convolution layer name.
        res_logger: Logger for results.
    """
    args_out = ['criterion']
    param = {}

    def __init__(self, args: Namespace, res_logger: ResLogger = None) -> None:
        r"""Assigning model identity.
        """
        self.model = args.model
        self.conv = args.conv
        self.model_repr, self.conv_repr = self.get_name(args)

        self.logger = logging.getLogger('log')
        self.res_logger = res_logger or ResLogger()

    @staticmethod
    def get_name(args: Namespace) -> tuple[str]:
        """Get model+conv name for logging path from argparse input without instantiation.
        Wrapper for :func:`pyg_spectral.nn.get_nn_name()`.

        Args:
            args: Configuration arguments.

                * args.model (str): Model architecture name.
                * args.conv (str): Convolution layer name.
                * other args specified in module :attr:`name` function.
        Returns:
            nn_name (tuple[str]): Name strings ``(model_name, conv_name)``.
        """
        return get_nn_name(args.model, args.conv, args)

    @staticmethod
    def get_trn(args: Namespace) -> TrnBase:
        r"""Get trainer class from model name.

        Args:
            args: Configuration arguments.
        Returns:
            trn (TrnBase): Trainer class.
        """
        model_repr, _ = get_nn_name(args.model, args.conv, args)
        return {
            'DecoupledFixed':   TrnFullbatch,
            'DecoupledVar':     TrnFullbatch,
            'Iterative':        TrnFullbatch,
            'IterativeFixed':   TrnFullbatch,
            'PrecomputedVar':   TrnMinibatch,
            'PrecomputedFixed': TrnMinibatch,
            'CppPrecFixed':     TrnMinibatch,
        }[model_repr]

    def _resolve_import(self, args: Namespace) -> tuple[str, str, dict]:
        class_name = self.model
        module_name = get_model_regi(self.model, 'module', args)
        kwargs = set_pargs(self.model, self.conv, args)

        # >>>>>>>>>>
        if module_name == 'torch_geometric.nn.models':
            args.criterion = 'BCEWithLogitsLoss' if args.out_channels == 1 else 'CrossEntropyLoss'

            del kwargs['conv']
            kwargs.setdefault('num_layers', kwargs.pop('num_hops'))
            kwargs.setdefault('dropout', kwargs.pop('dropout_lin'))

        elif module_name == 'pyg_spectral.nn.models_pyg':
            args.criterion = 'BCELoss' if args.out_channels == 1 else 'NLLLoss'

        # Default to load from `pyg_spectral.nn.models`
        else:
            args.criterion = 'BCELoss' if args.out_channels == 1 else 'NLLLoss'

            # Parse conv args
            for conv in self.conv.split(','):
                if conv in ['Adji2Conv', 'AdjSkip2Conv']:
                    kwargs['num_hops'] = int(kwargs['num_hops'] / 2)
            # Parse model args
        # <<<<<<<<<<
        return class_name, module_name, kwargs

    def get(self, args: Namespace) -> tuple[nn.Module, TrnBase]:
        r"""Load model with specified arguments.

        Args:
            args.num_hops (int): Number of conv hops.
            args.in_layers (int): Number of MLP layers before conv.
            args.out_layers (int): Number of MLP layers after conv.
            args.in_channels (int): Number of input features.
            args.out_channels (int): Number of output classes.
            args.hidden_channels (int): Number of hidden units.
            args.dropout_[lin/conv] (float): Dropout rate for linear/conv.
        Updates:
            args.criterion (str): Loss function name.
        """
        self.logger.debug('-'*20 + f" Loading model: {self} " + '-'*20)

        trn = self.get_trn(args)
        class_name, module_name, kwargs = self._resolve_import(args)
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
    def get(self, args: Namespace) -> tuple[nn.Module, TrnBase]:
        self.signature_lst = ['num_hops', 'in_layers', 'out_layers', 'hidden_channels', 'dropout_lin', 'dropout_conv']
        self.signature = {key: getattr(args, key) for key in self.signature_lst}

        trn = self.get_trn(args)
        class_name, module_name, kwargs = self._resolve_import(args)
        model = load_import(class_name, module_name)(**kwargs)
        if hasattr(model, 'reset_parameters'):
            model.reset_parameters()
        if hasattr(model, 'reset_cache'):
            model.reset_cache()

        self.res_logger.concat([('model', self.model), ('conv', self.conv_repr)])
        return model, trn

    def update(self, args: Namespace, model: nn.Module) -> nn.Module:
        signature = {key: getattr(args, key) for key in self.signature_lst}
        if self.signature != signature:
            self.signature = signature
            model, _ = self.get(args)
            return model

        if hasattr(model, 'reset_parameters'):
            model.reset_parameters()
        return model
