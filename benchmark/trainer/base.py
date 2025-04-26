# -*- coding:utf-8 -*-
"""
Author: nyLiao
File Created: 2024-03-03
"""
import logging
from argparse import Namespace

import torch
import torch.nn as nn
from torch_geometric.data import Data
import torch_geometric.utils as pyg_utils

from pyg_spectral import profile
from pyg_spectral.utils import load_import

from dataset import split_crossval
from utils import CkptLogger, ResLogger
from .load_metric import metric_loader


class TrnBase(object):
    r"""Base trainer class for general pipelines and tasks.

    Args:
        model: Pytorch model to be trained.
        data: PyG style data.
        res_logger: Logger for results.
        args: Configuration arguments.

            * device (str): torch device.
            * metric (str): Metric for evaluation.
            * criterion (set): Loss function in :mod:`torch.nn`.
            * epoch (int): Number of training epochs.
            * lr_[lin/conv] (float): Learning rate for linear/conv.
            * wd_[lin/conv] (float): Weight decay for linear/conv.
            * patience (int): Patience for early stopping.
            * period (int): Period for checkpoint saving.
            * suffix (str): Suffix for checkpoint saving.
            * storage (str): Storage scheme for checkpoint saving.
            * logpath (Path): Path for logging.
            * multi (bool): True for multi-label classification.
            * in_channels (int): Number of data input features.
            * out_channels (int): Number of data output classes.

    Methods:
        setup_optimizer: Set up the optimizer and scheduler.
        clear: Clear self cache.
        run: Run the training process.
    """
    name: str
    param = {
        'lr_lin':   ('float', (1e-5, 5e-1), {'log': True}, lambda x: float(f'{x:.3e}')),
        'lr_conv':  ('float', (1e-5, 5e-1), {'log': True}, lambda x: float(f'{x:.3e}')),
        'wd_lin':   ('float', (1e-7, 1e-3), {'log': True}, lambda x: float(f'{x:.3e}')),
        'wd_conv':  ('float', (1e-7, 1e-3), {'log': True}, lambda x: float(f'{x:.3e}')),
    }

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
        self.criterion = load_import(args.criterion, 'torch.nn')()
        self.data = data

        # Evaluation metrics
        self.multi = args.multi
        self.in_channels = args.in_channels
        self.out_channels = args.out_channels
        self.splits = ['train', 'val', 'test']
        metric = metric_loader(args).to(self.device)
        self.evaluator = {k: metric.clone(postfix='_'+k) for k in self.splits}

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
        self.logger.info(f"[trainer]: Loss: {self.criterion.__class__.__name__}, Optimizer: {self.optimizer.__class__.__name__}")

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
                  split_train: list[str] = ['train'],
                  split_val: list[str] = ['val']) -> ResLogger:
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
             split_test: list[str] = ['train', 'val', 'test']) -> ResLogger:
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

    def _learn_split(self, split: list[str] = ['train']) -> ResLogger:
        r"""Actual train iteration on the given splits."""
        raise NotImplementedError

    def _eval_split(self, split: list[str]) -> ResLogger:
        r"""Actual test on the given splits."""
        raise NotImplementedError

    def run(self) -> ResLogger:
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)


class TrnBase_Trial(TrnBase):
    r"""Trainer supporting optuna.pruners in training.
    """
    def __init__(self,
                 model: nn.Module,
                 data: Data,
                 args: Namespace,
                 res_logger: ResLogger = None,
                 **kwargs):
        super().__init__(model, data, args, res_logger, **kwargs)
        self.splits = ['train', 'val', 'hyperval', 'test']
        metric = metric_loader(args).to(self.device)
        self.evaluator = {k: metric.clone(postfix='_'+k) for k in self.splits}

    def split_hyperval(self, data: Data) -> Data:
        if hasattr(data, 'val_mask_neg'):
            # Actually index not mask
            label = torch.ones(data.val_mask.size(1), dtype=int)
            _, val_, hyperval_ = split_crossval(label, 1, data.val_mask.size(1)//2, ignore_neg=True, stratify=False, return_mask=False)
            data.hyperval_mask = data.val_mask[:, hyperval_]
            data.val_mask = data.val_mask[:, val_]

            label = torch.zeros(data.val_mask_neg.size(1), dtype=int)
            _, val_, hyperval_ = split_crossval(label, 1, data.val_mask_neg.size(1)//2, ignore_neg=True, stratify=False, return_mask=False)
            data.hyperval_mask_neg = data.val_mask_neg[:, hyperval_]
            data.val_mask_neg = data.val_mask_neg[:, val_]

        else:
            attr_to_index = lambda k: pyg_utils.mask_to_index(data[f'{k}_mask']) if hasattr(data, f'{k}_mask') else torch.tensor([])
            idx = {k: attr_to_index(k) for k in ['train', 'val', 'hyperval']}
            r_train = 1.0 * len(idx['train']) / (len(idx['train']) + len(idx['val']) + len(idx['hyperval']))
            r_val = r_hyperval = (1.0 - r_train) / 2

            label = data.y.detach().clone()
            label[data.test_mask] = -1
            data.train_mask, data.val_mask, data.hyperval_mask = split_crossval(label, r_train, r_val, ignore_neg=True, stratify=True)
        return data

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
                  split_train: list[str] = ['train'],
                  split_val: list[str] = ['val']) -> ResLogger:
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
