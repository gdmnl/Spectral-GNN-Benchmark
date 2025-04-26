# -*- coding:utf-8 -*-
"""Fullbatch, transductive node classification.
Author: nyLiao
File Created: 2024-02-26
"""
from argparse import Namespace

import torch
import torch.nn as nn
from torch_geometric.data import Data, Dataset
import torch_geometric.transforms as T
import torch_geometric.utils as pyg_utils

from pyg_spectral.profile import Stopwatch

from .base import TrnBase, TrnBase_Trial
from utils import ResLogger


class TrnFullbatch(TrnBase):
    r"""Fullbatch trainer class for node classification.
        - Model forward input: separate edge index and node features.
        - Run pipeline: train_val -> test.

    Args:
        model, data, res_logger: args for :class:`TrnBase`.
        args: args for :class:`TrnBase`.

            * device (str): torch device.
            * metric (str): Metric for evaluation.
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
    """
    name: str = 'fb'

    def __init__(self,
                 model: nn.Module,
                 data: Dataset,
                 args: Namespace,
                 **kwargs):
        super(TrnFullbatch, self).__init__(model, data, args, **kwargs)

        self.mask: dict = None
        self.flag_test_deg = args.test_deg if hasattr(args, 'test_deg') else False

    def clear(self):
        del self.mask, self.data
        return super().clear()

    def _fetch_data(self) -> tuple[Data, dict]:
        r"""Process the single graph data."""
        t_to_device = T.ToDevice(self.device, attrs=['x', 'y', 'adj_t', 'edge_index'] + [f'{k}_mask' for k in self.splits])
        self.data = t_to_device(self.data)
        # FIXME: Update to `EdgeIndex` [Release note 2.5.0](https://github.com/pyg-team/pytorch_geometric/releases/tag/2.5.0)
        # if not pyg_utils.is_sparse(self.data.adj_t):
        #     raise NotImplementedError
        # if pyg_utils.contains_isolated_nodes(self.data.edge_index):
        #     self.logger.warning(f"Graph {self.data} contains isolated nodes.")

        self.mask = {k: getattr(self.data, f'{k}_mask') for k in self.splits}
        return self.data, self.mask

    def _fetch_input(self) -> tuple:
        r"""Process each sample of model input and label."""
        if hasattr(self.data, 'adj_t'):
            input, label = (self.data.x, self.data.adj_t), self.data.y
        else:
            input, label = (self.data.x, self.data.edge_index), self.data.y
        if hasattr(self.model, 'preprocess'):
            self.model.preprocess(*input)
        return input, label

    # ===== Epoch run
    def _learn_split(self, split: list = ['train']) -> ResLogger:
        r"""Actual train iteration on the given splits."""
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
        r"""Actual test on the given splits."""
        self.model.eval()
        res = ResLogger()

        input, label = self._fetch_input()
        with Stopwatch() as stopwatch:
            output = self.model(*input)
            output = torch.sigmoid(output) if self.out_channels == 1 else torch.log_softmax(output, dim=-1)

        for k in split:
            mask_split = self.mask[k]
            self.evaluator[k](output[mask_split], label[mask_split])

            res.concat(self.evaluator[k].compute())
            self.evaluator[k].reset()

        return res.concat(
            [('time_eval', stopwatch.data)])

    # ===== Run block
    def test_deg(self) -> ResLogger:
        r"""Separate high/low degree subsets and evaluate."""
        import logging
        from torch_geometric.typing import SparseTensor
        RATIO_HIGH = 0.2

        adj_t = self.data.adj_t
        if isinstance(adj_t, SparseTensor):
            deg = adj_t.sum(dim=0).cpu()
        elif pyg_utils.is_torch_sparse_tensor(adj_t):
            deg = torch.sparse.sum(adj_t.to_sparse_coo(), [0]).cpu().to_dense()
        else:
            raise NotImplementedError(f"Type {type(adj_t)} not supported!")

        _, idx_high = torch.topk(deg, k=int(RATIO_HIGH * deg.size(0)))
        mask_high = torch.zeros_like(deg, dtype=torch.bool, device=self.device)
        mask_high[idx_high] = True

        self.mask['test_high'] = mask_high & self.mask['test']
        self.mask['test_low'] = (~mask_high) & self.mask['test']
        self.evaluator['test_high'] = self.evaluator['test'].clone(postfix='_high')
        self.evaluator['test_high'].reset()
        self.evaluator['test_low'] = self.evaluator['test'].clone(postfix='_low')
        self.evaluator['test_low'].reset()

        res_test = self._eval_split(['test_high', 'test_low'])
        res_test.concat([
            ('deg_high', self.mask['test_high'].sum().item()),
            ('deg_low', self.mask['test_low'].sum().item())])
        # self.logger.log(logging.LTRN, res_test.get_str())
        return res_test

    # ===== Run pipeline
    def run(self) -> ResLogger:
        res_run = ResLogger()
        self._fetch_data()
        self.model = self.model.to(self.device)
        self.setup_optimizer()

        res_train = self.train_val()
        res_run.merge(res_train)

        self.model = self.ckpt_logger.load('best', model=self.model)
        res_test = self.test()
        res_run.merge(res_test)

        if self.flag_test_deg:
            res_run.merge(self.test_deg())

        return self.res_logger.merge(res_run)


class TrnLPFullbatch(TrnFullbatch):
    name: str = 'lpfb'

    def _fetch_data(self) -> tuple[Data, dict]:
        r"""Process the single graph data."""
        t_to_device = T.ToDevice(self.device, attrs=['x', 'y', 'adj_t']
                                 + [f'{k}_mask' for k in self.splits] + [f'{k}_mask_neg' for k in self.splits])
        self.data = t_to_device(self.data)
        self.data.adj_t = self.data.adj_t.float()

        self.mask = {k: getattr(self.data, f'{k}_mask') for k in self.splits}
        self.mask_neg = {k: getattr(self.data, f'{k}_mask_neg') for k in self.splits}
        return self.data, self.mask, self.mask_neg

    def _fetch_input(self, split: list, train: bool=False) -> tuple:
        r"""Process each sample of model input and label."""
        assert hasattr(self.data, 'adj_t')

        edge_label, label = {}, {}
        for k in split:
            pos_edge_label = getattr(self.data, f'{k}_mask')
            neg_edge_label = getattr(self.data, f'{k}_mask_neg')
            if neg_edge_label.size(1) < 2 and train:
                # neg_edge_label = pyg_utils.negative_sampling(
                #     edge_index=self.data.edge_index,
                #     num_nodes=self.data.num_nodes,
                #     num_neg_samples=pos_edge_label.size(1), method='sparse').to(self.device)
                neg_edge_label = torch.randint(0, self.data.num_nodes, size=pos_edge_label.size(),
                                               dtype=torch.long, device=self.device)
            elif neg_edge_label.size(1) < 2:
                neg_edge_label = torch.empty((2, 0), dtype=torch.long, device=self.device)

            edge_label[k] = torch.cat([pos_edge_label, neg_edge_label], dim=1)
            label[k] = torch.cat([torch.ones(pos_edge_label.size(1), dtype=torch.float, device=self.device),
                                  torch.zeros(neg_edge_label.size(1), dtype=torch.float, device=self.device)], dim=0)

        input = (self.data.x, self.data.adj_t)

        if hasattr(self.model, 'preprocess'):
            self.model.preprocess(*input)
        return input, edge_label, label

    # ===== Epoch run
    def _learn_split(self, split: list = ['train']) -> ResLogger:
        r"""Actual train iteration on the given splits."""
        assert len(split) == 1
        self.model.train()

        input, edge_label, label = self._fetch_input(split, train=True)
        with Stopwatch() as stopwatch:
            self.optimizer.zero_grad()
            output = self.model(*input)
            output = self.model.decode(output, edge_label[split[0]])

            loss = self.criterion(output, label[split[0]])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.desc_mlp.parameters(), 1.0)
            self.optimizer.step()

        return ResLogger()(
            [('time_learn', stopwatch.data),
             (f'loss_{split[0]}', loss.item())])

    @torch.no_grad()
    def _eval_split(self, split: list = ['test']) -> ResLogger:
        r"""Actual test on the given splits."""
        self.model.eval()
        res = ResLogger()

        input, edge_label, label = self._fetch_input(split, train=False)
        with Stopwatch() as stopwatch:
            output = self.model(*input)

        for k in split:
            with stopwatch:
                output_k = self.model.decode(output, edge_label[k])
                output_k = torch.sigmoid(output_k)
            self.evaluator[k](output_k, label[k])

            res.concat(self.evaluator[k].compute())
            self.evaluator[k].reset()

        return res.concat(
            [('time_eval', stopwatch.data)])


class TrnFullbatch_Trial(TrnFullbatch, TrnBase_Trial):
    r"""Trainer supporting optuna.pruners in training.
    """
    def run(self) -> ResLogger:
        res_run = ResLogger()
        self.data = self.split_hyperval(self.data)
        self._fetch_data()
        self.model = self.model.to(self.device)
        self.setup_optimizer()

        res_train = self.train_val()
        res_run.merge(res_train)

        self.model = self.ckpt_logger.load('best', model=self.model)
        res_test = self.test(['train', 'val', 'hyperval', 'test'])
        res_run.merge(res_test)

        return self.res_logger.merge(res_run)

    def update(self, *args, **kwargs):
        self.__init__(*args, **kwargs)
        return self
