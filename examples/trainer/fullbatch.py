# -*- coding:utf-8 -*-
"""Fullbatch, node classification.
Author: nyLiao
File Created: 2024-02-26
"""
import logging
from argparse import Namespace
import torch
import torch.nn as nn
from torch_geometric.data import Data, Dataset
import torch_geometric.transforms as T
import torch_geometric.utils as pyg_utils

from pyg_spectral.profile import Stopwatch, Accumulator

from .base import TrnBase
from utils import ResLogger


class TrnFullbatchIter(TrnBase):
    r"""Fullbatch trainer class for node classification.
        - Model forward input: separate edge index and node features.
        - Run pipeline: train_val -> test.
    """
    def __init__(self,
                 model: nn.Module,
                 dataset: Dataset,
                 args: Namespace,
                 **kwargs):
        super(TrnFullbatchIter, self).__init__(model, dataset, args, **kwargs)
        self.mask: dict = None
        self.data: Data = None

    def clear(self):
        del self.mask, self.data
        return super().clear()

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

        with Stopwatch() as stopwatch:
            self.optimizer.zero_grad()
            output = self.model(*input)
            mask_split = self.mask[split[0]]
            loss = self.criterion(output[mask_split], label[mask_split])
            loss.backward()
            self.optimizer.step()

        return ResLogger()(
            [('time_learn', stopwatch.data),
             (f'loss_{split[0]}', loss.item())])

    @torch.no_grad()
    def _eval_split(self, split: list = ['test']) -> ResLogger:
        self.model.eval()
        input, label = self._fetch_input()

        with Stopwatch() as stopwatch:
            output = self.model(*input)

        res = ResLogger()
        for k in split:
            mask_split = self.mask[k]
            self.evaluator[k](output[mask_split], label[mask_split])
            res.concat(self.evaluator[k].compute())
            self.evaluator[k].reset()

        return res.concat(
            [('time_eval', stopwatch.data)])

    # ===== Run block
    @TrnBase._log_memory(split='train')
    def train_val(self) -> ResLogger:
        self.logger.debug('-'*20 + f" Start training: {self.epoch} " + '-'*20)

        # TODO: list of accumulators
        time_learn = Accumulator()
        res_learn = ResLogger()
        for epoch in range(1, self.epoch+1):
            res_learn.concat([('epoch', epoch, lambda x: format(x, '03d'))], row=epoch)

            res = self._learn_split()
            res_learn.merge(res, rows=[epoch])
            time_learn.update(res_learn[epoch, 'time_learn'])

            res = self._eval_split(['val'])
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

    def train_val_test(self) -> ResLogger:
        """Run test in every training epoch, do not need to save checkpoint."""
        # res_val = self._eval_split(['train', 'val', 'test'])
        raise NotImplementedError

    @TrnBase._log_memory(split='eval')
    def test(self) -> ResLogger:
        self.logger.debug('-'*20 + f" Start evaluating: train+val+test " + '-'*20)

        res_test = self._eval_split(['train', 'val', 'test'])
        return res_test

    # ===== Run pipeline
    def run(self) -> ResLogger:
        res_run = ResLogger()
        self.model = self.model.to(self.device)
        self._fetch_data()
        self.setup_optimizer()

        res_train = self.train_val()
        res_run.merge(res_train)

        self.model = self.ckpt_logger.load('best', model=self.model)
        res_test = self.test()
        res_run.merge(res_test)

        return self.res_logger.merge(res_run)


# TODO: possible to decouple model.conv on CPU?
# class TrnFullbatchDec(TrnBase):
#     pass
