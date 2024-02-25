# -*- coding:utf-8 -*-
"""
Author: nyLiao
File Created: 2023-03-20
File: training.py
"""
from typing import Union, Callable
from pathlib import Path

import os
import uuid
import random
import copy

import numpy as np
import torch
import torch.nn as nn


# noinspection PyUnresolvedReferences
def set_seed(seed: int = None, cuda: bool = True):
    if seed is None:
        seed = int(uuid.uuid4().hex, 16) % 1000000
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)
    return seed


class CkptLogger(object):
    r"""Checkpoint Logger for saving and loading models and managing early
    stopping during training.

    Args:
        logpath (Path): The path to the directory where the checkpoints will be saved.
        patience (int, optional): Patience number for early stopping. Defaults no early stopping.
        period (int, optional): Periodic saving interval. Defaults to no periodic saving.
        prefix (str, optional): Prefix for the checkpoint file names.
        storage (str, optional): Storage scheme for saving the checkpoints.
            * 'model' vs 'state': Save model object or state_dict.
            * 'ram', 'gpu': Save as file, RAM or GPU memory.
        metric_cmp (function or ['max', 'min'], optional): Comparison function for the metric.
    """
    def __init__(self,
                 logpath: Path,
                 patience: int = 999999,
                 period: int = 0,
                 prefix: str = 'model',
                 storage: str = 'model_gpu',
                 metric_cmp: Union[Callable[[float, float], bool], str]='max'):
        self.logpath = logpath
        self.patience = patience
        self.period = period
        self.prefix = prefix
        # Checkpoint storage scheme
        assert storage in ['model', 'state', 'model_ram', 'state_ram', 'model_gpu', 'state_gpu']
        self.storage = storage
        # Comparison function for metric
        if isinstance(metric_cmp, str):
            assert metric_cmp in ['max', 'min']
            self.cmp = lambda x, y: x > y if metric_cmp == 'max' else x < y
        else:
            self.cmp = metric_cmp

        self.epoch_current = 0
        self.epoch_from_best = 0
        self.epoch_best = 0
        self.metric_best = None

    def set_epoch(self, epoch: int):
        self.epoch_current = epoch

    # ===== Checkpoint file IO
    def _get_model_file(self, *suffix) -> Path:
        return self.logpath.joinpath(f'{self.prefix}_{"-".join(suffix)}.pth')

    def get_last_epoch(self) -> int:
        if self.epoch_current == 0:
            for file in self.logpath.glob(f'{self.prefix}_*.pth'):
                suffix = file.stem.split('_')[1:]
                epoch = int(suffix[0]) if suffix[0].isdigit() else 0
                if epoch > self.epoch_current:
                    self.epoch_current = epoch
        return self.epoch_current

    def save(self, *suffix, model: nn.Module):
        r"""Save the model according to storage scheme.

        Args:
            suffix: Variable length argument for suffix in the model file name.
            model (nn.Module): The model to be saved.
        """
        name = f'{self.prefix}_{"-".join(suffix)}'
        path = self._get_model_file(*suffix)

        if self.storage == 'state':
            torch.save(model.state_dict(), path)
        elif self.storage == 'model':
            torch.save(model, path)
        elif self.storage == 'state_gpu':
            # Alternative way is to use BytesIO
            if hasattr(self, name): delattr(self, name)
            self.setattr(name, copy.deepcopy(model.state_dict()))
        elif self.storage == 'state_ram':
            if hasattr(self, name): delattr(self, name)
            model_copy = copy.deepcopy(self.state_dict)
            model_copy = {k: v.cpu() for k, v in model_copy.items()}
            self.setattr(name, model_copy)
        elif self.storage == 'model_gpu':
            if hasattr(self, name): delattr(self, name)
            self.setattr(name, copy.deepcopy(model))
        elif self.storage == 'model_ram':
            # TODO: reduce mem fro 2xmem(model) to mem(model)
            if hasattr(self, name): delattr(self, name)
            device = next(model.parameters()).device
            self.setattr(name, copy.deepcopy(model.cpu()))
            model.to(device)

    def load(self, *suffix, model: nn.Module, map_location='cpu') -> nn.Module:
        r"""Load the model from the storage.

        Args:
            suffix: Variable length argument for suffix in the model file name.
            model (nn.Module): The model structure to load.
            map_location (str, optional): `map_location` argument for `torch.load`.

        Returns:
            model (nn.Module): The loaded model.
        """
        name = f'{self.prefix}_{"-".join(suffix)}'
        path = self._get_model_file(*suffix)

        if self.storage == 'state':
            state_dict = torch.load(path, map_location=map_location)
            model.load_state_dict(state_dict)
        elif self.storage in ['state_ram', 'state_gpu']:
            assert hasattr(self, name)
            model.load_state_dict(getattr(self, name))
            # model.to(map_location)
        elif self.storage == 'model':
            model = torch.load(path, map_location=map_location)
        elif self.storage in ['model_ram', 'model_gpu']:
            assert hasattr(self, name)
            model = copy.deepcopy(getattr(self, name))
            # model.to(map_location)

        return model

    # ===== Early stopping
    @property
    def is_early_stop(self) -> bool:
        return self.epoch_from_best >= self.patience

    def step(self,
             metric: float,
             model: nn.Module = None) -> bool:
        """Step one epoch with periodic saving and early stopping.

        Args:
            metric (float): Metric value for the current step.
            model (nn.Module, optional): Model for the current step. Defaults to None.

        Returns:
            early_stop (bool): True if early stopping criteria is met.
        """
        self.epoch_current += 1
        if self.metric_best is None or self.cmp(metric, self.metric_best):
            self.epoch_best = self.epoch_current
            self.metric_best = metric
            self.epoch_from_best = 0
            if model is not None:
                self.save('best', model=model)
        else:
            self.epoch_from_best += 1

        if self.period > 0 and self.epoch_current % self.period == 0:
            if model is not None:
                self.save(str(self.epoch_current), model=model)

        return self.is_early_stop
