# -*- coding:utf-8 -*-
"""Full batch, node classification.
    - Model forward input: separate edge index and node features.
    - Run pipeline: train_val -> test.
Author: nyLiao
File Created: 2024-02-26
File: fb_iter.py
"""
from typing import List
from logging import Logger
from argparse import Namespace

import torch
import torch.nn as nn
from torch_geometric.data import Data, Dataset
import torch_geometric.transforms as T
import torch_geometric.utils as pyg_utils

from pyg_spectral import metrics

from utils import CkptLogger, ResLogger


LTRN = 15


class TrnBase(object):
    r"""Base trainer class for general pipelines and tasks.

    Args:
        model (nn.Module): Pytorch model to be trained.
        dataset (Dataset): PyG style dataset.
        logger (Logger): Logger object.
        args (Namespace): Configuration arguments.

    Methods:
        setup_optimizer: Set up the optimizer and scheduler.
        clear: Clear self cache.
        run: Run the training process.
    """
    def __init__(self,
                 model: nn.Module,
                 dataset: Dataset,
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

        self.splits = ['train', 'val', 'test']
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes

        self.logger = logger
        # assert (args.quiet or '_file' not in storage), "Storage scheme cannot be file for quiet run."
        self.ckpt_logger = CkptLogger(
            self.logpath,
            patience=self.patience,
            period=self.period,
            prefix=(f'model-{self.suffix}' if self.suffix else 'model'),
            storage='state_gpu')

    def setup_optimizer(self):
        # TODO: layer-specific wd [no wd for theta](https://github.com/seijimaekawa/empirical-study-of-GNNs/blob/main/models/train_model.py#L204)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5,
            threshold=1e-4, patience=15, verbose=False)

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

    def clear(self):
        del self.model
        del self.dataset
        del self.optimizer
        del self.scheduler

    def run(self) -> ResLogger:
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)


class TrnFullbatchIter(TrnBase):
    def __init__(self,
                 model: nn.Module,
                 dataset: Dataset,
                 logger: Logger,
                 args: Namespace,
                 **kwargs):
        super(TrnFullbatchIter, self).__init__(model, dataset, logger, args, **kwargs)
        self.mask: dict = None
        self.data: Data = None

    def _fetch_data(self) -> Data:
        t_to_device = T.ToDevice(self.device, attrs=['x', 'y', 'adj_t'])
        self.data = t_to_device(self.dataset[0])

        # FIXME: Update to `EdgeIndex` [Release note 2.5.0](https://github.com/pyg-team/pytorch_geometric/releases/tag/2.5.0)
        if not pyg_utils.is_sparse(self.data.adj_t):
            raise NotImplementedError
        # if pyg_utils.contains_isolated_nodes(self.data.edge_index):
        #     self.logger.warning(f"Graph {self.data} contains isolated nodes.")

        self.mask = {k: getattr(self.data, f'{k}_mask') for k in self.splits}
        self.logger.info(f"[data]: {self.data}")
        return self.data

    def _fetch_input(self) -> tuple:
        input, label = (self.data.x, self.data.adj_t), self.data.y
        return input, label

    # ===== Epoch run
    def _learn_split(self, split: list = ['train']) -> ResLogger:
        assert len(split) == 1
        self.model.train()
        input, label = self._fetch_input()

        stopwatch = metrics.Stopwatch()

        stopwatch.start()
        self.optimizer.zero_grad()
        output = self.model(*input)
        mask_split = self.mask[split[0]]
        loss = self.criterion(output[mask_split], label[mask_split])
        loss.backward()
        self.optimizer.step()
        stopwatch.pause()

        return ResLogger()(
            [('time_learn', stopwatch.time),
             (f'loss_{split[0]}', loss.item())])

    def _eval_split(self, split: list = ['test']) -> ResLogger:
        self.model.eval()
        input, label = self._fetch_input()

        # TODO: more metrics: Calcualtor -> Evaluator
        calc = {k: metrics.F1Calculator(self.num_classes) for k in split}
        stopwatch = metrics.Stopwatch()

        with torch.no_grad():
            stopwatch.start()
            output = self.model(*input)
            stopwatch.pause()

            output = output.cpu().detach()
            output = output.argmax(dim=1)
            label = label.cpu().detach()
            for k in split:
                mask_split = self.mask[k]
                calc[k].step(label[mask_split], output[mask_split])

        return ResLogger()(
            [(f'metric_{k}', calc[k].get('micro')) for k in split] +
            [('time_eval', stopwatch.time)])

    # ===== Run block
    def train_val(self) -> ResLogger:
        self.logger.debug('-'*20 + f" Start training: {self.epoch} " + '-'*20)
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

            self.logger.log(LTRN, res_learn.get_str(row=epoch))

            self.ckpt_logger.step(metric_val, self.model)
            self.ckpt_logger.set_best(epoch_best=epoch)
            if self.ckpt_logger.is_early_stop:
                break

        res_train = ResLogger()
        res_train.concat(self.ckpt_logger.get_best())
        res_train.concat(
            [('epoch', self.ckpt_logger.epoch_current),
             ('time', time_learn.data),
             ('mem_rem', metrics.MemoryRAM()(unit='G')),
             ('mem_cuda', metrics.MemoryCUDA()(unit='G')),],
            suffix='learn')
        return res_train

    def train_val_test(self) -> ResLogger:
        """Run test in every training epoch, do not need to save checkpoint."""
        # res_val = self._eval_split(['train', 'val', 'test'])
        raise NotImplementedError

    def test(self) -> ResLogger:
        self.logger.debug('-'*20 + f" Start evaluating: train+val+test " + '-'*20)
        res_test = self._eval_split(['train', 'val', 'test'])

        return res_test.concat(
            [('mem_rem', metrics.MemoryRAM()(unit='G')),
             ('mem_cuda', metrics.MemoryCUDA()(unit='G')),],
            suffix='eval')

    # ===== Run pipeline
    def run(self) -> ResLogger:
        res_run = ResLogger()
        self.model = self.model.to(self.device)
        self._fetch_data()
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


# TODO: possible to decouple model.conv on CPU?
# class TrnFullbatchDec(TrnBase):
#     pass
