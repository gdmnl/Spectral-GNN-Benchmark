# -*- coding:utf-8 -*-
"""
Author: nyLiao
File Created: 2024-02-26
"""
from pathlib import Path
from argparse import Namespace
import logging
import uuid

import torch
import torch_geometric
from torch_geometric.data import Data
import torch_geometric.transforms as T
import pyg_spectral.transforms as Tspec

from utils import ResLogger
from dataset import dataset_map, f_get_data, T_insert


DATAPATH = Path('../data')


class SingleGraphLoader(object):
    r"""Loader for :class:`torch_geometric.data.Data` object for one graph.

    Args:
        args: Configuration arguments.

            * args.seed (int): Random seed.
            * args.data (str): Dataset name.
            * args.data_split (str): Index of dataset split.
        res_logger: Logger for results.
    """
    args_out = ['in_channels', 'out_channels', 'multi', 'metric']
    param = {
        'normg': ('float', (0.0, 1.0), {'step': 0.05}, lambda x: round(x, 2)),
    }

    def __init__(self, args: Namespace, res_logger: ResLogger = None) -> None:
        # Assigning dataset identity.
        self.seed = args.seed
        self.data = args.data.lower()

        # Prevent using `edge_index` for more [Memory-Efficient Computation](https://pytorch-geometric.readthedocs.io/en/latest/notes/sparse_tensor.html)
        # assert torch_geometric.typing.WITH_TORCH_SPARSE
        # FIXME: check NormalizeFeatures
        if self.data in ['ogbl-collab', ]:
            self.transform = T.Compose([
                T.RemoveIsolatedNodes(),
                T.RemoveDuplicatedEdges(reduce='mean'),
                T.ToUndirected(),
                # T.AddRemainingSelfLoops(fill_value=1.0),
                Tspec.RemoveSelfLoops(),
                T.ToSparseTensor(remove_edge_index=True, layout=torch.sparse_csr),  # torch.sparse.Tensor
            ])
        else:
            self.transform = T.Compose([
                T.RemoveIsolatedNodes(),
                T.RemoveDuplicatedEdges(reduce='mean'),
                # T.LargestConnectedComponents(),
                T.AddRemainingSelfLoops(fill_value=1.0),
                T.NormalizeFeatures(),
                # T.ToSparseTensor(remove_edge_index=True),                         # torch_sparse.SparseTensor
                T.ToSparseTensor(remove_edge_index=True, layout=torch.sparse_csr),  # torch.sparse.Tensor
            ])

        self.logger = logging.getLogger('log')
        self.res_logger = res_logger or ResLogger()

    @staticmethod
    def available_datasets() -> dict:
        return dataset_map

    def get(self, args: Namespace) -> Data:
        r"""Load data based on parameters.

        Args:
            args.normg (float): Generalized graph norm.

        Updates:
            args.in_channels (int): Number of input features.
            args.out_channels (int): Number of output classes.
            args.multi (bool): True for multi-label classification.
            args.metric (str): Main metric name for evaluation.
        """
        self.logger.debug('-'*20 + f" Loading data: {self} " + '-'*20)

        T_insert(self.transform, Tspec.GenNorm(left=args.normg), index=-2)
        # get_data(datapath, transform, args)
        data = f_get_data[dataset_map[self.data]](DATAPATH, self.transform, args)

        self.logger.info(f"[dataset]: {self.data} (features={args.in_channels}, classes={args.out_channels}, metric={args.metric})")
        self.logger.info(f"[data]: {data}")
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
        self.signature = {key: getattr(args, key) for key in self.signature_lst}
        self.random_state = args.seed
        args.data_split += f"_{args.seed}"

        T_insert(self.transform, Tspec.GenNorm(left=args.normg), index=-2)
        data = f_get_data[dataset_map[self.data]](DATAPATH, self.transform, args)

        self.res_logger.concat([('data', self.data, str), ('metric', args.metric, str)])
        return data

    def update(self, args: Namespace, data: Data) -> Data:
        r"""Update data split for the next trial.
        """
        signature = {key: getattr(args, key) for key in self.signature_lst}
        if self.signature != signature:
            self.signature = signature
            self.random_state = uuid.uuid5(uuid.NAMESPACE_DNS, str(self.random_state)).int % 2**32
            args.data_split = '_'.join(args.data_split.split('_')[:2] + [str(self.random_state)])

            self.transform.transforms[-2] = Tspec.GenNorm(left=args.normg)
            data = f_get_data[dataset_map[self.data]](DATAPATH, self.transform, args)

        return data
