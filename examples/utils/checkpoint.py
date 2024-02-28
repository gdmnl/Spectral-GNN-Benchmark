# -*- coding:utf-8 -*-
"""
Author: nyLiao
File Created: 2023-03-20
File: checkpoint.py
"""
from typing import Union, Callable
from pathlib import Path

import copy
import torch
import torch.nn as nn


class CkptLogger(object):
    r"""Checkpoint Logger for saving and loading models and managing early
    stopping during training.

    Args:
        logpath (Path or str): Path to checkpoints saving directory.
        patience (int, optional): Patience for early stopping. Defaults no early stopping.
        period (int, optional): Periodic saving interval. Defaults to no periodic saving.
        prefix (str, optional): Prefix for the checkpoint file names.
        storage (str, optional): Storage scheme for saving the checkpoints.
            - 'model' vs 'state': Save model object or state_dict.
            - '_file', '_ram', '_gpu': Save as file, RAM, or GPU memory.
        metric_cmp (function or ['max', 'min'], optional): Comparison function for the metric.
    """
    def __init__(self,
                 logpath: Union[Path, str],
                 patience: int = -1,
                 period: int = 0,
                 prefix: str = 'model',
                 storage: str = 'state_gpu',
                 metric_cmp: Union[Callable[[float, float], bool], str]='max'):
        self.logpath = Path(logpath)
        self.prefix = prefix
        self.filetype = 'pth'
        self.patience = patience
        self.period = period
        # Checkpoint storage scheme
        assert storage in ['model_file', 'state_file', 'model_ram', 'state_ram', 'model_gpu', 'state_gpu']
        self.storage = storage
        # Comparison function for metric
        if isinstance(metric_cmp, str):
            assert metric_cmp in ['max', 'min']
            self.cmp = lambda x, y: (x > y) if metric_cmp == 'max' else (x < y)
        else:
            self.cmp = metric_cmp

        self.set_epoch()

    def set_epoch(self, epoch: int = 0):
        self.epoch_current = epoch
        self.epoch_from_best = 0
        self.metric_best = None
        self.last_improved = False

    # ===== Checkpoint file IO
    def _get_model_path(self, *suffix) -> Path:
        return self.logpath.joinpath(f'{self.prefix}_{"-".join(suffix)}.{self.filetype}')

    def get_last_epoch(self) -> int:
        r"""Get last saved model epoch. Useful for deciding load model path."""
        if self.epoch_current == 0:
            for file in self.logpath.glob(f'{self.prefix}_*.{self.filetype}'):
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
        path = self._get_model_path(*suffix)

        if self.storage == 'state_file':
            torch.save(model.state_dict(), path)
        elif self.storage == 'model_file':
            torch.save(model, path)
        elif self.storage == 'state_gpu':
            # Alternative way is to use BytesIO
            if hasattr(self, name): delattr(self, name)
            setattr(self, name, copy.deepcopy(model.state_dict()))
        elif self.storage == 'state_ram':
            if hasattr(self, name): delattr(self, name)
            model_copy = copy.deepcopy(self.state_dict)
            model_copy = {k: v.cpu() for k, v in model_copy.items()}
            setattr(self, name, model_copy)
        elif self.storage == 'model_gpu':
            if hasattr(self, name): delattr(self, name)
            setattr(self, name, copy.deepcopy(model))
        elif self.storage == 'model_ram':
            # FIXME: reduce mem fro 2xmem(model) to mem(model)
            if hasattr(self, name): delattr(self, name)
            device = next(model.parameters()).device
            setattr(self, name, copy.deepcopy(model.cpu()))
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
        path = self._get_model_path(*suffix)

        if self.storage == 'state_file':
            state_dict = torch.load(path, map_location=map_location)
            model.load_state_dict(state_dict)
        elif self.storage in ['state_ram', 'state_gpu']:
            assert hasattr(self, name)
            model.load_state_dict(getattr(self, name))
            # model.to(map_location)
        elif self.storage == 'model_file':
            model = torch.load(path, map_location=map_location)
        elif self.storage in ['model_ram', 'model_gpu']:
            assert hasattr(self, name)
            model = copy.deepcopy(getattr(self, name))
            # model.to(map_location)

        # NOTE: aggressively remove stored model memory
        self.clear()
        return model

    def clear(self):
        for attr in list(self.__dict__):
            if attr.startswith(self.prefix):
                delattr(self, attr)

    # ===== Early stopping
    @property
    def is_early_stop(self) -> bool:
        r"""Whether current epoch satisfies early stopping criteria."""
        if self.patience < 0:
            return False
        return self.epoch_from_best >= self.patience

    @property
    def is_period(self) -> bool:
        r"""Whether current epoch should do periodic saving."""
        return self.period > 0 and self.epoch_current % self.period == 0

    def _is_improved(self, metric) -> bool:
        r"""Whether the metric is better than previous best."""
        return self.metric_best is None or self.cmp(metric, self.metric_best)

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
        if self.is_period:
            if model is not None:
                self.save(str(self.epoch_current), model=model)

        self.last_improved = self._is_improved(metric)
        if self.last_improved:
            self.metric_best = metric
            self.epoch_from_best = 0
            if model is not None:
                self.save('best', model=model)
        else:
            self.epoch_from_best += 1

        return self.last_improved

    def set_best(self, **kwargs):
        r"""Save given args to model attributes if is the best epoch."""
        if not self.last_improved:
            return
        self.best_keys = kwargs.keys()
        for key, val in kwargs.items():
            setattr(self, key, val)

    def get_best(self) -> list:
        r"""Get saved model attributes from the best epoch."""
        return [(key, getattr(self, key)) for key in self.best_keys]
