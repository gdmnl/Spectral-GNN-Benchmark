# -*- coding:utf-8 -*-
"""
Author: nyLiao
File Created: 2024-02-26
File: load_model.py
"""
from argparse import Namespace
import torch.nn as nn

from pyg_spectral.utils import load_import


class LoaderModel(object):
    def __init__(self) -> None:
        r"""For entities."""
        pass

    def get(self,
            model: str,
            conv: str,
            args: Namespace) -> nn.Module:
        kwargs = dict(
            conv=conv,
            in_channels=args.num_features,
            out_channels=args.num_classes,
            hidden_channels=args.hidden,
            num_layers=args.layer,)
        lib_model = 'pyg_spectral.nn.models'
        fload = lambda kw: load_import(model, lib_model)(**kw)

        if model in ['IterConv']:
            if conv in ['FixLinSumAdj', 'VarLinSumAdj']:
                kwargs.update(dict(
                    theta=('appr', 0.15),
                    dropout=args.dp,
                    K=2,))
                return fload(kwargs)
        elif model in ['DecPostMLP']:
            if conv in ['FixSumAdj', 'VarSumAdj']:
                kwargs.update(dict(
                    theta=('appr', 0.15),
                    dropout=args.dp,
                    K=5,))
                return fload(kwargs)

        raise ValueError(f"Model '{model}:{conv}' not found.")

    def __call__(self, *args, **kwargs):
        return self.get(*args, **kwargs)
