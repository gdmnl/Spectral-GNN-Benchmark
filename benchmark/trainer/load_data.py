# -*- coding:utf-8 -*-
"""
Author: nyLiao
File Created: 2024-02-26
"""
from pathlib import Path
from argparse import Namespace
import logging

import torch
import torch_geometric
from torch_geometric.data import Data
import torch_geometric.transforms as T

import pyg_spectral.transforms as Tspec
from pyg_spectral.utils import load_import

from utils import ResLogger
from dataset import class_list, func_list


DATAPATH = Path('../data')


class SingleGraphLoader(object):
    r"""Loader for PyG.data.Data object for one graph.

    Args:
        args.seed (int): Random seed.
        args.data (str): Dataset name.
        args.data_split (str): Index of dataset split.
    """
    def __init__(self, args: Namespace, res_logger: ResLogger = None) -> None:
        r"""Assigning dataset identity.
        """
        self.seed = args.seed
        self.data = args.data.lower()

        self.logger = logging.getLogger('log')
        self.res_logger = res_logger or ResLogger()

    def get(self, args: Namespace) -> Data:
        r"""Load data based on parameters.

        Args:
            args.normg (float): Generalized graph norm.

        Returns (update in args):
            args.num_features (int): Number of input features.
            args.num_classes (int): Number of output classes.
            args.multi (bool): True for multi-label classification.
            args.metric (str): Main metric name for evaluation.
        """
        self.logger.debug('-'*20 + f" Loading data: {self} " + '-'*20)

        # Prevent using `edge_index` for more [Memory-Efficient Computation](https://pytorch-geometric.readthedocs.io/en/latest/notes/sparse_tensor.html)
        # assert torch_geometric.typing.WITH_TORCH_SPARSE
        # FIXME: check NormalizeFeatures
        self.transform = T.Compose([
            T.RemoveIsolatedNodes(),
            T.RemoveDuplicatedEdges(reduce='mean'),
            # T.LargestConnectedComponents(),
            T.AddRemainingSelfLoops(fill_value=1.0),
            T.NormalizeFeatures(),
            Tspec.GenNorm(left=args.normg),
            # T.ToSparseTensor(remove_edge_index=True),                         # torch_sparse.SparseTensor
            T.ToSparseTensor(remove_edge_index=True, layout=torch.sparse_csr),  # torch.sparse.Tensor
        ])
        assert self.data in class_list, f"Invalid dataset: {self.data}"
        get_data = func_list[class_list[self.data]]
        data = get_data(DATAPATH, self.transform, args)

        self.logger.info(f"[dataset]: {self.data} (features={args.num_features}, classes={args.num_classes})")
        self.logger.info(f"[data]: {data}")
        self.logger.info(f"[metric]: {args.metric}")
        split_dict = {k[:-5]: v.sum().item() for k, v in data.items() if k.endswith('_mask')}
        self.logger.info(f"[split]: {args.data_split} {split_dict}")
        self.res_logger.concat([('data', self.data, str), ('metric', args.metric, str)])
        return data

    def __call__(self, *args, **kwargs):
        return self.get(*args, **kwargs)

    def __str__(self) -> str:
        return f"{self.data}"


class SingleGraphLoader_Trial(SingleGraphLoader):
    r"""Reuse necessary data for multiple runs.
    """
    def get(self, args: Namespace) -> Data:
        self.signature_lst = ['normg']
        self.signature = {key: args.__dict__[key] for key in self.signature_lst}

        module_name, class_name, kwargs, metric = self._resolve_import(args)
        self.dataset = load_import(class_name, module_name)(**kwargs)
        data = self.dataset[0]
        data = self._resolve_split(self.dataset, data)

        self._get_properties(self.dataset, data)
        args.num_features, args.num_classes = self.num_features, self.num_classes
        args.multi = False
        args.metric = self.metric = metric

        # Remaining resolvers
        if not args.multi and data.y.dim() > 1 and data.y.size(1) == 1:
            data.y = data.y.flatten()

        self.res_logger.concat([('data', self.data, str), ('metric', metric, str)])
        return data, metric

    def update(self, args: Namespace, data: Data) -> Data:
        r"""Update data split for the next trial.
        """
        signature = {key: args.__dict__[key] for key in self.signature_lst}
        if self.signature != signature:
            self.transform.transforms[-2] = Tspec.GenNorm(left=args.normg)
            data = self.dataset[0]
            data = self._resolve_split(self.dataset, data)

        args.num_features, args.num_classes = self.num_features, self.num_classes
        args.multi = False
        args.metric = self.metric
        if not args.multi and data.y.dim() > 1 and data.y.size(1) == 1:
            data.y = data.y.flatten()
        return data
