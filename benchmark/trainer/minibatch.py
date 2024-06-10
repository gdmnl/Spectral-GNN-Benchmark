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

from .base import TrnBase, TrnBase_Trial
from .load_metric import metric_loader
from utils import CkptLogger, ResLogger


class TrnMinibatch(TrnBase):
    r"""Minibatch trainer class for node classification.
        - Model forward input: node embeddings.
        - Run pipeline: propagate -> train_val -> test.

    Args:
        args.batch (int): Batch size.
        args.normf (int): Embedding normalization.
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
    name: str = 'mb'

    def __init__(self,
                 model: nn.Module,
                 data: Dataset,
                 args: Namespace,
                 **kwargs):
        super(TrnMinibatch, self).__init__(model, data, args, **kwargs)
        self.batch = args.batch
        if args.normf is not None:
            assert isinstance(args.normf, int)
            self.norm_prop = TensorStandardScaler(dim=args.normf)

        self.shuffle = {'train': True, 'val': False, 'test': False}
        self.embed = None

    def clear(self):
        del self.data, self.embed
        return super().clear()

    def _fetch_data(self) -> Tuple[Data, dict]:
        r"""Process the single graph data."""
        # FIXME: Update to `EdgeIndex` [Release note 2.5.0](https://github.com/pyg-team/pytorch_geometric/releases/tag/2.5.0)
        if not pyg_utils.is_sparse(self.data.adj_t):
            raise NotImplementedError
        # if pyg_utils.contains_isolated_nodes(self.data.edge_index):
        #     self.logger.warning(f"Graph {self.data} contains isolated nodes.")

        mask = {k: getattr(self.data, f'{k}_mask') for k in self.splits}
        return self.data, mask

    def _fetch_preprocess(self, data: Data) -> tuple:
        r"""Call model preprocess for precomputation."""
        input, label = (data.x, data.adj_t), data.y
        if hasattr(self.model, 'preprocess'):
            self.model.preprocess(*input)
        return input, label

    def _fetch_input(self, split: str) -> Generator:
        r"""Process each sample of model input and label for training."""
        # for input, label in self.embed[split]:
        #     yield input.to(self.device), label.to(self.device)
        # =====
        if self.shuffle[split]:
            idxs = torch.randperm(len(self.embed[split]))
        else:
            idxs = torch.arange(len(self.embed[split]))
        for idx in idxs.split(self.batch):
            input, label = self.embed[split][idx]
            yield input.to(self.device), label.to(self.device)

    # ===== Epoch run
    def _learn_split(self, split: list = ['train']) -> ResLogger:
        r"""Actual train iteration on the given splits."""
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
        r"""Actual test on the given splits."""
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
    def preprocess(self) -> ResLogger:
        r"""Pipeline for precomputation on CPU."""
        self.logger.debug('-'*20 + f" Start propagation: pre " + '-'*20)

        data, mask = self._fetch_data()
        input, label = self._fetch_preprocess(data)
        del self.data

        stopwatch = Stopwatch()
        if hasattr(self.model, 'convolute'):
            with stopwatch:
                embed = self.model.convolute(*input)
        else:
            embed = input[0]

        if hasattr(self, 'norm_prop'):
            self.norm_prop.fit(embed[mask['train']])
            embed = self.norm_prop(embed)

        self.embed = {}
        for k in self.splits:
            dataset = TensorDataset(embed[mask[k]], label[mask[k]])
            # self.embed[k] = DataLoader(dataset,
            #                           batch_size=self.batch,
            #                           shuffle=self.shuffle[k],
            #                           num_workers=0)
            self.embed[k] = dataset
            self.logger.log(logging.LTRN, f"[{k}]: n_sample={len(dataset)}, n_batch={len(self.embed[k]) // self.batch}")
        return ResLogger()(
            [('time_pre', stopwatch.data)])

    # ===== Run pipeline
    def run(self) -> ResLogger:
        res_run = ResLogger()

        self.model = self.model.cpu()
        res_pre = self.preprocess()
        res_run.merge(res_pre)

        if hasattr(self.model, 'reset_cache'):
            self.model.reset_cache()
        self.model = self.model.to(self.device)
        self.setup_optimizer()

        res_train = self.train_val()
        res_run.merge(res_train)

        self.model = self.ckpt_logger.load('best', model=self.model)
        res_test = self.test()
        res_run.merge(res_test)

        return self.res_logger.merge(res_run)


class TrnMinibatch_Trial(TrnMinibatch, TrnBase_Trial):
    r"""Trainer supporting optuna.pruners in training.
    Lazy calling precomputation.
    """
    def run(self) -> ResLogger:
        res_run = ResLogger()

        if self.embed is None:
            self.model = self.model.cpu()
            res_pre = self.preprocess()
        else:
            res_pre = ResLogger()(
                [('time_pre', 0), ('mem_ram_pre', 0), ('mem_cuda_pre', 0)])
        res_run.merge(res_pre)

        if hasattr(self.model, 'reset_cache'):
            self.model.reset_cache()
        self.model = self.model.to(self.device)
        self.setup_optimizer()

        res_train = self.train_val()
        res_run.merge(res_train)

        self.model = self.ckpt_logger.load('best', model=self.model)
        res_test = self.test()
        res_run.merge(res_test)

        return self.res_logger.merge(res_run)

    def update(self,
                 model: nn.Module,
                 data: Data,
                 args: Namespace,
                 res_logger: ResLogger = None,
                 **kwargs):
        self.model = model
        self.data = data
        self.res_logger = res_logger or ResLogger()
        metric = metric_loader(args).to(self.device)
        self.evaluator = {k: metric.clone(postfix='_'+k) for k in self.splits}
        self.ckpt_logger = CkptLogger(
            self.logpath,
            patience=self.patience,
            period=self.period,
            prefix=('-'.join(filter(None, ['model', self.suffix]))),
            storage=self.storage)

        self.signature_lst = ['normg', 'alpha', 'beta', 'theta_param']
        signature = {key: args.__dict__[key] for key in self.signature_lst}
        if not hasattr(self, 'signature') or self.signature != signature:
            self.signature = signature
            self.embed = None
