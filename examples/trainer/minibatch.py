# -*- coding:utf-8 -*-
"""Minibatch, node classification.
Author: nyLiao
File Created: 2024-03-03
"""
from typing import Tuple, Generator
import logging
from argparse import Namespace

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.data import Data, Dataset
import torch_geometric.transforms as T
import torch_geometric.utils as pyg_utils

from pyg_spectral.profile import Stopwatch, Accumulator

from .base import TrnBase
from utils import ResLogger


class TrnMinibatchDec(TrnBase):
    r"""Minibatch trainer class for node classification.
        - Model forward input: node embeddings.
        - Run pipeline: propagate -> train_val -> test.
    """
    def __init__(self,
                 model: nn.Module,
                 dataset: Dataset,
                 args: Namespace,
                 **kwargs):
        super(TrnMinibatchDec, self).__init__(model, dataset, args, **kwargs)
        self.batch = args.batch

        self.shuffle = {'train': True, 'val': False, 'test': False}
        self.data = None

    def clear(self):
        del self.data
        return super().clear()

    def _fetch_data(self) -> Tuple[Data, dict]:
        data = self.dataset[0]
        # FIXME: Update to `EdgeIndex` [Release note 2.5.0](https://github.com/pyg-team/pytorch_geometric/releases/tag/2.5.0)
        if not pyg_utils.is_sparse(data.adj_t):
            raise NotImplementedError
        # if pyg_utils.contains_isolated_nodes(self.data.edge_index):
        #     self.logger.warning(f"Graph {self.data} contains isolated nodes.")
        self.logger.info(f"[data]: {data}")

        mask = {k: getattr(data, f'{k}_mask') for k in self.splits}
        split_dict = {k: v.sum().item() for k, v in mask.items()}
        self.logger.info(f"[split]: {split_dict}")

        return data, mask

    def _fetch_input_propagate(self, data: Data) -> tuple:
        input, label = (data.x, data.adj_t), data.y
        return input, label

    def _fetch_input(self, split: str) -> Generator:
        for input, label in self.data[split]:
            yield input.to(self.device), label.to(self.device)

    # ===== Epoch run
    def _learn_split(self, split: list = ['train']) -> ResLogger:
        assert len(split) == 1
        self.model.train()
        loss_epoch = Accumulator()
        stopwatch = Stopwatch()

        for it, (input, label) in enumerate(self._fetch_input(split[0])):
            with stopwatch:
                self.optimizer.zero_grad()
                output = self.model(input)
                loss = self.criterion(output, label)
                loss.backward()
                self.optimizer.step()

            loss_epoch.update(loss.item(), count=label.size(0))

        return ResLogger()(
            [('time_learn', stopwatch.data),
             (f'loss_{split[0]}', loss_epoch.mean)])

    @torch.no_grad()
    def _eval_split(self, split: list = ['test']) -> ResLogger:
        self.model.eval()
        stopwatch = Stopwatch()
        res = ResLogger()

        for k in split:
            for it, (input, label) in enumerate(self._fetch_input(k)):
                with stopwatch:
                    output = self.model(input)

                self.evaluator[k](output, label)

            res.concat(self.evaluator[k].compute())
            res.concat([('time', stopwatch.data)], suffix=k)
            self.evaluator[k].reset()
            stopwatch.reset()

        return res

    # ===== Run block
    @TrnBase._log_memory(split='pre')
    def propagate(self) -> dict:
        data, mask = self._fetch_data()
        input, label = self._fetch_input_propagate(data)
        del self.dataset

        embed = self.model.propagate(*input)
        # TODO: embed normalization

        self.data = {}
        for k in self.splits:
            dataset = TensorDataset(embed[mask[k]], label[mask[k]])
            self.data[k] = DataLoader(dataset,
                                      batch_size=self.batch,
                                      shuffle=self.shuffle[k],
                                      num_workers=0)
        return ResLogger()

    # ===== Run pipeline
    def run(self) -> ResLogger:
        res_run = ResLogger()

        # TODO: trainable parameters in propagate
        res_pre = self.propagate()
        res_run.merge(res_pre)

        self.model = self.model.to(self.device)
        self.setup_optimizer()

        res_train = self.train_val()
        res_run.merge(res_train)

        self.model = self.ckpt_logger.load('best', model=self.model)
        res_test = self.test()
        res_run.merge(res_test)

        return self.res_logger.merge(res_run)
