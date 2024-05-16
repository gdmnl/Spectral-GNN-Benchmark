# -*- coding:utf-8 -*-
"""Minibatch with random sampling for precompute models, transductive node classification.
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
import torch_geometric.utils as pyg_utils

from pyg_spectral.nn.norm import TensorStandardScaler
from pyg_spectral.profile import Stopwatch, Accumulator

from .base import TrnBase
from utils import ResLogger


class TrnMinibatch(TrnBase):
    r"""Minibatch trainer class for node classification.
        - Model forward input: node embeddings.
        - Run pipeline: propagate -> train_val -> test.

    Args:
        args.batch (int): Batch size.
        args.normf (int): Embedding normalization.
    """
    name: str = 'mb'

    def __init__(self,
                 model: nn.Module,
                 dataset: Dataset,
                 args: Namespace,
                 **kwargs):
        super(TrnMinibatch, self).__init__(model, dataset, args, **kwargs)
        self.batch = args.batch
        if args.normf is not None:
            assert isinstance(args.normf, int)
            self.norm_prop = TensorStandardScaler(dim=args.normf)

        self.shuffle = {'train': True, 'val': False, 'test': False}
        self.embed = None

    def clear(self):
        del self.data
        return super().clear()

    def _fetch_data(self) -> Tuple[Data, dict]:
        data = self.data
        # FIXME: Update to `EdgeIndex` [Release note 2.5.0](https://github.com/pyg-team/pytorch_geometric/releases/tag/2.5.0)
        if not pyg_utils.is_sparse(data.adj_t):
            raise NotImplementedError
        # if pyg_utils.contains_isolated_nodes(self.data.edge_index):
        #     self.logger.warning(f"Graph {self.data} contains isolated nodes.")

        mask = {k: getattr(data, f'{k}_mask') for k in self.splits}
        return data, mask

    def _fetch_preprocess(self, data: Data) -> tuple:
        input, label = (data.x, data.adj_t), data.y
        if hasattr(self.model, 'preprocess'):
            self.model.preprocess(*input)
        return input, label

    def _fetch_input(self, split: str) -> Generator:
        for input, label in self.embed[split]:
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
    def preprocess(self) -> dict:
        self.logger.debug('-'*20 + f" Start propagation: pre " + '-'*20)

        data, mask = self._fetch_data()
        input, label = self._fetch_preprocess(data)
        del self.data

        embed = self.model.convolute(*input)
        if self.norm_prop:
            self.norm_prop.fit(embed[mask['train']])
            embed = self.norm_prop(embed)

        self.embed = {}
        for k in self.splits:
            dataset = TensorDataset(embed[mask[k]], label[mask[k]])
            self.embed[k] = DataLoader(dataset,
                                      batch_size=self.batch,
                                      shuffle=self.shuffle[k],
                                      num_workers=0)
            self.logger.log(logging.LTRN, f"[{k}]: n_sample={len(dataset)}, n_batch={len(self.data[k])}")
        return ResLogger()

    # ===== Run pipeline
    def run(self) -> ResLogger:
        raise DeprecationWarning
        res_run = ResLogger()

        # TODO: check behavior of trainable parameters in propagate
        res_pre = self.preprocess()
        res_run.merge(res_pre)

        self.model = self.model.to(self.device)
        self.setup_optimizer()

        res_train = self.train_val()
        res_run.merge(res_train)

        self.model = self.ckpt_logger.load('best', model=self.model)
        res_test = self.test()
        res_run.merge(res_test)

        return self.res_logger.merge(res_run)
