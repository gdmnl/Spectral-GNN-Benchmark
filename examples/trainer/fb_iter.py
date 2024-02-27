# -*- coding:utf-8 -*-
"""
Author: nyLiao
File Created: 2024-02-26
File: fb_iter.py
"""
from logging import Logger
from argparse import Namespace

import torch
import torch.nn as nn
from torch_geometric.utils import is_sparse
from torch_geometric.data import Data, InMemoryDataset
from pyg_spectral import metrics

from utils import CkptLogger, ResLogger


class TrnBase(object):
    def __init__(self,
                 model: nn.Module,
                 dataset: InMemoryDataset,
                 logger: Logger,
                 args: Namespace,
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

        self.logger = logger
        # assert (args.quiet or '_file' not in storage), "Storage scheme cannot be file for quiet run."
        self.ckpt_logger = CkptLogger(
            self.logpath,
            patience=self.patience,
            period=self.period,
            prefix=(f'model-{self.suffix}' if self.suffix else 'model'),
            storage='state_gpu')

    def setup_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5,
            threshold=1e-4, patience=15, verbose=False)

    def _fetch_data(self) -> Data:
        r"""Dataset to data."""
        raise NotImplementedError

    def _fetch_input(self, data: Data) -> tuple:
        r"""Data to model input and label."""
        raise NotImplementedError

    def _learn_split(self, split):
        raise NotImplementedError

    def _eval_split(self, split):
        raise NotImplementedError

    def clear(self):
        del self.model
        del self.dataset
        del self.optimizer
        del self.scheduler

    def run(self):
        raise NotImplementedError


class TrnFullbatchIter(TrnBase):
    def __init__(self,
                 model: nn.Module,
                 dataset: InMemoryDataset,
                 logger: Logger,
                 args: Namespace,
                 **kwargs):
        super(TrnFullbatchIter, self).__init__(model, dataset, logger, args, **kwargs)
        self.data = None

    def _fetch_data(self) -> Data:
        data = self.dataset[0].to(self.device)
        self.logger.debug(f"[data]:{data}")
        return data

    def _fetch_input(self, data: Data) -> tuple:
        # FIXME: Update to `EdgeIndex` [Release note 2.5.0](https://github.com/pyg-team/pytorch_geometric/releases/tag/2.5.0)
        if hasattr(data, 'adj_t') and is_sparse(data.adj_t):
            input, label = (data.x, data.adj_t), data.y
        else:
            raise NotImplementedError
        return input, label

    def _learn_split(self, split: list = ['train']) -> ResLogger:
        assert len(split) == 1
        self.model.train()
        input, label = self._fetch_input(self.data)

        stopwatch = metrics.Stopwatch()

        stopwatch.start()
        self.optimizer.zero_grad()
        output = self.model(*input)
        mask_split = getattr(self.data, f'{split[0]}_mask')
        loss = self.criterion(output[mask_split], label[mask_split])
        loss.backward()
        self.optimizer.step()
        stopwatch.pause()

        return ResLogger().concat(
            [('time_learn', stopwatch.time),
             (f'loss_{split[0]}', loss.item())])

    def _eval_split(self, split: list = ['test']) -> ResLogger:
        self.model.eval()
        input, label = self._fetch_input(self.data)

        # TODO: more metrics: Calcualtor -> Evaluator
        calc = {k: metrics.F1Calculator(self.dataset.num_classes) for k in split}
        stopwatch = metrics.Stopwatch()

        with torch.no_grad():
            stopwatch.start()
            output = self.model(*input)
            stopwatch.pause()

            output = output.cpu().detach()
            output = output.argmax(dim=1)
            label = label.cpu().detach()
            for k in split:
                mask_split = getattr(self.data, f'{k}_mask')
                calc[k].update(label[mask_split], output[mask_split])

        return ResLogger().concat(
            [(f'metric_{k}', calc[k].get('micro')) for k in split] +
            [('time_eval', stopwatch.time)])

    def train_val(self) -> ResLogger:
        # TODO: list of accumulators
        time_learn = metrics.Accumulator()
        res_learn = ResLogger()

        for epoch in range(1, self.epoch+1):
            res_learn.concat([('epoch', epoch)], row=epoch)

            res_train = self._learn_split()
            res_learn.merge(res_train, rows=[epoch])
            time_learn.update(res_learn[epoch, 'time_learn'])

            res_val = self._eval_split(['val'])
            res_learn.merge(res_val, rows=[epoch])
            metric_val = res_learn[epoch, 'metric_val']
            self.scheduler.step(metric_val)

            self.logger.info(res_learn.get_str(row=epoch))

            self.ckpt_logger.step(metric_val, self.model)
            self.ckpt_logger.set_best(epoch_best=epoch, metric_best=metric_val)
            if self.ckpt_logger.is_early_stop:
                break

        res_train = ResLogger()
        res_train.concat(self.ckpt_logger.get_best())
        res_train.concat(
            [('epoch', self.ckpt_logger.epoch_current),
             ('time', time_learn.data),
             ('mem_rem', metrics.MemoryRAM().get(unit='G')),
             ('mem_cuda', metrics.MemoryCUDA().get(unit='G')),],
            suffix='learn')
        return res_train

    def train_val_test(self) -> ResLogger:
        """Run test in every training epoch, do not need to save checkpoint."""
        # res_val = self._eval_split(['train', 'val', 'test'])
        raise NotImplementedError

    def test(self) -> ResLogger:
        res_test = self._eval_split(['train', 'val', 'test'])

        return res_test.concat(
            [('mem_rem', metrics.MemoryRAM().get(unit='G')),
             ('mem_cuda', metrics.MemoryCUDA().get(unit='G')),],
            suffix='eval')

    def run(self) -> ResLogger:
        res_run = ResLogger()
        self.model = self.model.to(self.device)
        self.data = self._fetch_data()
        self.setup_optimizer()

        with self.device:
            torch.cuda.empty_cache()
        res_train = self.train_val()
        res_run.merge(res_train)

        with self.device:
            torch.cuda.empty_cache()
        self.model = self.ckpt_logger.load('best', model=self.model)
        res_test = self.test()
        res_run.merge(res_test)

        return res_run
