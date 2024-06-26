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
from torch_geometric.data import Data
from pyg_spectral import profile

from utils import CkptLogger, ResLogger


class TrnBase(object):
    r"""Base trainer class for general pipelines and tasks.

    Args:
        model (nn.Module): Pytorch model to be trained.
        data (Data): PyG style data.
        logger (Logger): Logger object.
        args (Namespace): Configuration arguments.
            device (str): torch device.
            metric (str): Metric for evaluation.
            epoch (int): Number of training epochs.
            lr_[lin/conv] (float): Learning rate for linear/conv.
            wd_[lin/conv] (float): Weight decay for linear/conv.
            patience (int): Patience for early stopping.
            period (int): Period for checkpoint saving.
            suffix (str): Suffix for checkpoint saving.
            storage (str): Storage scheme for checkpoint saving.
            logpath (Path): Path for logging.
            multi (bool): True for multi-label classification.
            num_features (int): Number of data input features.
            num_classes (int): Number of data output classes.

    Methods:
        setup_optimizer: Set up the optimizer and scheduler.
        clear: Clear self cache.
        run: Run the training process.
    """
    name: str

    def __init__(self,
                 model: nn.Module,
                 data: Data,
                 args: Namespace,
                 res_logger: ResLogger = None,
                 **kwargs):
        # Get args
        self.device = args.device
        self.epoch = args.epoch
        self.optimizer_dct = {
            'lin':  {'lr': args.lr_lin,  'weight_decay': args.wd_lin},
            'conv': {'lr': args.lr_conv, 'weight_decay': args.wd_conv},}
        self.patience = args.patience
        self.period = args.period
        self.logpath = args.logpath
        self.suffix = args.suffix
        self.storage = args.storage

        # Get entities
        self.model = model
        self.data = data

        # Evaluation metrics
        self.splits = ['train', 'val', 'test']
        self.multi = args.multi
        self.num_features = args.num_features
        self.num_classes = args.num_classes

        # Loggers
        self.logger = logging.getLogger('log')
        self.res_logger = res_logger or ResLogger()
        # assert (args.quiet or '_file' not in storage), "Storage scheme cannot be file for quiet run."
        self.ckpt_logger = CkptLogger(
            self.logpath,
            patience=self.patience,
            period=self.period,
            prefix=('-'.join(filter(None, ['model', self.suffix]))),
            storage=self.storage)
        self.metric_ckpt = args.metric + '_val'

    def setup_optimizer(self):
        if hasattr(self.model, 'get_optimizer'):
            self.optimizer = torch.optim.Adam(
                self.model.get_optimizer(self.optimizer_dct))
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), **self.optimizer_dct['lin'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5,
            threshold=1e-4, patience=15)

    def clear(self):
        if self.evaluator:
            for k in self.splits:
                if self.evaluator[k]:
                    self.evaluator[k].reset()
                    del self.evaluator[k]
            del self.evaluator
        if self.scheduler: del self.scheduler
        if self.optimizer: del self.optimizer
        if self.model: del self.model
        if self.data: del self.data

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

    # ===== Run block
    @_log_memory(split='train')
    def train_val(self,
                  split_train: List[str] = ['train'],
                  split_val: List[str] = ['val']) -> ResLogger:
        r"""Pipeline for iterative training.

        Args:
            split_train (list): Training splits.
            split_val (list): Validation splits.
        """
        self.logger.debug('-'*20 + f" Start training: {self.epoch} " + '-'*20)

        time_learn = profile.Accumulator()
        res_learn = ResLogger()
        for epoch in range(1, self.epoch+1):
            res_learn.concat([('epoch', epoch, lambda x: format(x, '03d'))], row=epoch)

            res = self._learn_split(split_train)
            res_learn.merge(res, rows=[epoch])
            time_learn.update(res_learn[epoch, 'time_learn'])

            # Set split_val=['train','val','test'] to run test in every training epoch, and no need to save checkpoint.
            res = self._eval_split(split_val)
            res_learn.merge(res, rows=[epoch])
            metric_val = res_learn[epoch, self.metric_ckpt]
            self.scheduler.step(metric_val)

            self.logger.log(logging.LTRN, res_learn.get_str(row=epoch))

            self.ckpt_logger.step(metric_val, self.model)
            self.ckpt_logger.set_at_best(epoch_best=epoch)
            if self.ckpt_logger.is_early_stop:
                break

        res_train = ResLogger()
        res_train.concat(self.ckpt_logger.get_at_best())
        res_train.concat(
            [('epoch', self.ckpt_logger.epoch_current),
             ('time', time_learn.data),],
            suffix='learn')
        return res_train

    @_log_memory(split='eval')
    def test(self,
             split_test: List[str] = ['train', 'val', 'test']) -> ResLogger:
        r"""Pipeline for testing.
        Args:
            split_test (list): Testing splits.
        """
        self.logger.debug('-'*20 + f" Start evaluating: {'+'.join(split_test)} " + '-'*20)

        res_test = self._eval_split(split_test)
        return res_test

    # ===== Run helpers
    def _fetch_data(self):
        r"""Process the single graph data."""
        raise NotImplementedError

    def _fetch_input(self) -> tuple:
        r"""Process each sample of model input and label."""
        raise NotImplementedError

    def _learn_split(self, split: List[str] = ['train']) -> ResLogger:
        r"""Actual train iteration on the given splits."""
        raise NotImplementedError

    def _eval_split(self, split: List[str]) -> ResLogger:
        r"""Actual test on the given splits."""
        raise NotImplementedError

    def run(self) -> ResLogger:
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)


class TrnBase_Trial(TrnBase):
    r"""Trainer supporting optuna.pruners in training.
    """
    def clear(self):
        if self.evaluator:
            for k in self.splits:
                if self.evaluator[k]:
                    self.evaluator[k].reset()
                    del self.evaluator[k]
            del self.evaluator
        if self.scheduler: del self.scheduler
        if self.optimizer: del self.optimizer

    def train_val(self,
                  split_train: List[str] = ['train'],
                  split_val: List[str] = ['val']) -> ResLogger:
        import optuna

        time_learn = profile.Accumulator()
        res_learn = ResLogger()
        for epoch in range(1, self.epoch+1):
            res_learn.concat([('epoch', epoch, lambda x: format(x, '03d'))], row=epoch)

            res = self._learn_split(split_train)
            res_learn.merge(res, rows=[epoch])
            time_learn.update(res_learn[epoch, 'time_learn'])

            res = self._eval_split(split_val)
            res_learn.merge(res, rows=[epoch])
            metric_val = res_learn[epoch, self.metric_ckpt]
            self.scheduler.step(metric_val)

            self.logger.log(logging.LTRN, res_learn.get_str(row=epoch))

            self.ckpt_logger.step(metric_val, self.model)
            self.ckpt_logger.set_at_best(epoch_best=epoch)
            if self.ckpt_logger.is_early_stop:
                break

            self.trial.report(metric_val, epoch-1)
            if self.trial.should_prune():
                raise optuna.TrialPruned()

        res_train = ResLogger()
        res_train.concat(self.ckpt_logger.get_at_best())
        res_train.concat(
            [('epoch', self.ckpt_logger.epoch_current),
             ('time', time_learn.data),],
            suffix='learn')
        return res_train
