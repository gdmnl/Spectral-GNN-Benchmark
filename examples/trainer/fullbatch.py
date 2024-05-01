# -*- coding:utf-8 -*-
"""Fullbatch, node classification.
Author: nyLiao
File Created: 2024-02-26
"""
from typing import Tuple
from argparse import Namespace
import os
import torch
import torch.nn as nn
from torch_geometric.data import Data, Dataset
import torch_geometric.transforms as T
import torch_geometric.utils as pyg_utils

from pyg_spectral.profile import Stopwatch

from .base import TrnBase
from utils import ResLogger, tsne_plt


class TrnFullbatchIter(TrnBase):
    r"""Fullbatch trainer class for node classification.
        - Model forward input: separate edge index and node features.
        - Run pipeline: train_val -> test.
    """
    name: str = 'fb'

    def __init__(self,
                 model: nn.Module,
                 dataset: Dataset,
                 args: Namespace,
                 **kwargs):
        super(TrnFullbatchIter, self).__init__(model, dataset, args, **kwargs)
        self.mask: dict = None
        self.data: Data = None
        self.args = args

    def clear(self):
        del self.mask, self.data
        return super().clear()

    def _fetch_data(self) -> Tuple[Data, dict]:
        t_to_device = T.ToDevice(self.device, attrs=['x', 'y', 'adj_t', 'train_mask', 'val_mask', 'test_mask'])
        self.data = t_to_device(self.dataset[0])
        # FIXME: Update to `EdgeIndex` [Release note 2.5.0](https://github.com/pyg-team/pytorch_geometric/releases/tag/2.5.0)
        if not pyg_utils.is_sparse(self.data.adj_t):
            raise NotImplementedError
        # if pyg_utils.contains_isolated_nodes(self.data.edge_index):
        #     self.logger.warning(f"Graph {self.data} contains isolated nodes.")
        self.logger.info(f"[data]: {self.data}")

        self.mask = {k: getattr(self.data, f'{k}_mask') for k in self.splits}
        split_dict = {k: v.sum().item() for k, v in self.mask.items()}
        
        self.logger.info(f"[split]: {split_dict}")

        return self.data, self.mask

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
        import logging

        adj_t = self.data.adj_t
        deg = adj_t.sum(dim=0).cpu()
        _, idx_high = torch.topk(deg, k=int(0.2*deg.size(0)))
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
            ('num_high', self.mask['test_high'].sum().item()),
            ('num_low', self.mask['test_low'].sum().item())])
        self.logger.log(logging.LTRN, res_test.get_str())
        return res_test
    
    

    # ===== Run tsne
    def draw_tsne(self) -> ResLogger:
        self.input, self.label = self._fetch_input()
        def feature_visualization_hook(module, input, output):
            print("input", input)
            print("output", output)
            tsne_plt(input[0].detach(), self.label, save_path=os.path.join(self.args.logpath, self.args.conv+ "_optimal/input.jpg"))
            tsne_plt(output.detach(), self.label, save_path=os.path.join(self.args.logpath, self.args.conv+ "_optimal/output.jpg"))
            
            # return input

        self.model.eval()
        res = ResLogger()
        if self.args.model in ['PostMLP']:
            hook = self.model.mlp.register_forward_hook(feature_visualization_hook)
        elif self.model in ['PreDecMLP']:
            hook = self.model.mlp.register_forward_hook(feature_visualization_hook)
        elif self.model in ['PreMLP']:
            hook = self.model.hiddenmlp.register_forward_hook(feature_visualization_hook)
        with Stopwatch() as stopwatch:
            output = self.model(*self.input)


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

        res_run.merge(self.test_deg())
        if self.tsne:
           self.draw_tsne()

        return self.res_logger.merge(res_run)


# FIXME: possible to decouple model.conv on CPU?
# class TrnFullbatchDec(TrnBase):
#     pass
