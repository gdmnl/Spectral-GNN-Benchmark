# -*- coding:utf-8 -*-
"""
Author: nyLiao
File Created: 2024-03-03
"""
from typing import List
import logging
from argparse import Namespace

import torch
import torch.nn as nn
from torch_geometric.data import Data, Dataset

from pyg_spectral import profile

from .load_metric import metric_loader
from utils import CkptLogger, ResLogger


LTRN = 15


class TrnBase(object):
    r"""Base trainer class for general pipelines and tasks.

    Args:
        model (nn.Module): Pytorch model to be trained.
        dataset (Dataset): PyG style dataset.
        logger (Logger): Logger object.
        args (Namespace): Configuration arguments.
            device (str): torch device.
            epoch (int): Number of training epochs.
            lr (float): Learning rate.
            wd (float): Weight decay.
            patience (int): Patience for early stopping.
            period (int): Period for checkpoint saving.
            suffix (str): Suffix for checkpoint saving.
            logpath (Path): Path for logging.
            multi (bool): True for multi-label classification.
            num_features (int): Number of dataset input features.
            num_classes (int): Number of dataset output classes.

    Methods:
        setup_optimizer: Set up the optimizer and scheduler.
        clear: Clear self cache.
        run: Run the training process.
    """
    def __init__(self,
                 model: nn.Module,
                 dataset: Dataset,
                 args: Namespace,
                 res_logger: ResLogger = None,
                 **kwargs):
        # Get args
        self.device = args.device
        self.epoch = args.epoch
        self.lr = args.lr
        self.wd = args.wd
        self.patience = args.patience
        self.period = args.period
        self.logpath = args.logpath
        self.suffix = args.suffix

        # Get entities
        self.model = model
        self.dataset = dataset
        self.criterion = nn.CrossEntropyLoss()

        # Evaluation metrics
        self.splits = ['train', 'val', 'test']
        self.multi = args.multi
        self.num_features = args.num_features
        self.num_classes = args.num_classes
        metric = metric_loader(args)
        self.evaluator = {k: metric.clone(postfix='_'+k) for k in self.splits}

        # Loggers
        self.logger = logging.getLogger('log')
        self.res_logger = res_logger or ResLogger()
        # assert (args.quiet or '_file' not in storage), "Storage scheme cannot be file for quiet run."
        self.ckpt_logger = CkptLogger(
            self.logpath,
            patience=self.patience,
            period=self.period,
            prefix=(f'model-{self.suffix}' if self.suffix else 'model'),
            storage='state_gpu')
        self.metric_ckpt = 'fimacro_val' if self.multi else 'f1micro_val'

    def setup_optimizer(self):
        # TODO: layer-specific wd [no wd for theta](https://github.com/seijimaekawa/empirical-study-of-GNNs/blob/main/models/train_model.py#L204)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5,
            threshold=1e-4, patience=15, verbose=False)

    def clear(self):
        del self.model
        del self.dataset
        del self.optimizer
        del self.scheduler
        for k in self.evaluator:
            del self.evaluator[k]

    @staticmethod
    def _log_memory(split: str = None, row: int = 0):
        def decorator(func):
            def wrapper(self, *args, **kwargs):
                with self.device:
                    torch.cuda.empty_cache()

                res = func(self, *args, **kwargs)
                res.concat(
                    [('mem_ram', profile.MemoryRAM()(unit='G')),
                     ('mem_cuda', profile.MemoryCUDA()(unit='G')),],
                    row=row, suffix=split)

                with self.device:
                    torch.cuda.empty_cache()
                return res
            return wrapper
        return decorator

    def _fetch_data(self) -> Data:
        r"""Dataset to single graph data."""
        raise NotImplementedError

    def _fetch_input(self, data: Data) -> tuple:
        r"""Data to model input and label."""
        raise NotImplementedError

    def _learn_split(self, split: List[str]) -> ResLogger:
        raise NotImplementedError

    def _eval_split(self, split: List[str]) -> ResLogger:
        raise NotImplementedError

    def run(self) -> ResLogger:
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)
