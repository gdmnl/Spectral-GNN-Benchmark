# -*- coding:utf-8 -*-
"""Fullbatch, transductive node classification.
Author: nyLiao
File Created: 2024-02-26
"""
from typing import Tuple
from argparse import Namespace

import torch
import torch.nn as nn
from torch_geometric.data import Data, Dataset
import torch_geometric.transforms as T
import torch_geometric.utils as pyg_utils

from pyg_spectral.profile import Stopwatch

from .base import TrnBase
from .load_metric import metric_loader
from utils import ResLogger


class TrnFullbatch(TrnBase):
    r"""Fullbatch trainer class for node classification.
        - Model forward input: separate edge index and node features.
        - Run pipeline: train_val -> test.

    Args:
        --- TrnBase Args ---
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
    """
    name: str = 'fb'

    def __init__(self,
                 model: nn.Module,
                 data: Dataset,
                 args: Namespace,
                 **kwargs):
        super(TrnFullbatch, self).__init__(model, data, args, **kwargs)
        metric = metric_loader(args).to(self.device)
        self.evaluator = {k: metric.clone(postfix='_'+k) for k in self.splits}
        self.criterion = nn.BCELoss() if self.num_classes == 1 else nn.NLLLoss()

        self.mask: dict = None
        self.flag_test_deg = args.test_deg if hasattr(args, 'test_deg') else False

    def clear(self):
        del self.mask, self.data
        return super().clear()

    def _fetch_data(self) -> Tuple[Data, dict]:
        r"""Process the single graph data."""
        t_to_device = T.ToDevice(self.device, attrs=['x', 'y', 'adj_t', 'train_mask', 'val_mask', 'test_mask'])
        self.data = t_to_device(self.data)
        # FIXME: Update to `EdgeIndex` [Release note 2.5.0](https://github.com/pyg-team/pytorch_geometric/releases/tag/2.5.0)
        if not pyg_utils.is_sparse(self.data.adj_t):
            raise NotImplementedError
        # if pyg_utils.contains_isolated_nodes(self.data.edge_index):
        #     self.logger.warning(f"Graph {self.data} contains isolated nodes.")

        self.mask = {k: getattr(self.data, f'{k}_mask') for k in self.splits}
        return self.data, self.mask

    def _fetch_input(self) -> tuple:
        r"""Process each sample of model input and label."""
        input, label = (self.data.x, self.data.adj_t), self.data.y
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
        elif isinstance(adj_t, torch.Tensor) and adj_t.is_sparse_csr:
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
