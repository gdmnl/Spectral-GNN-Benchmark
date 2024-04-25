# -*- coding:utf-8 -*-
"""
Author: nyLiao
File Created: 2024-02-26
"""
from typing import List
from pathlib import Path
from argparse import Namespace
import logging

import torch
import torch_geometric
from torch_geometric.data import Dataset
import torch_geometric.transforms as T
from torch_geometric.data.dataset import _get_flattened_data_list
import pyg_spectral.transforms as Tspec
from pyg_spectral.utils import load_import

from utils import ResLogger


DATAPATH = Path('../data')


class DatasetLoader(object):
    r"""Loader for PyG.data.Dataset object.

    Args:
        args.data (str): Dataset name.
        args.split_idx (int): Index of dataset split.
    """
    def __init__(self, args: Namespace, res_logger: ResLogger = None) -> None:
        r"""Assigning dataset identity.
        """
        self.data = args.data.lower()
        self.split_idx = args.split_idx

        self.logger = logging.getLogger('log')
        self.res_logger = res_logger or ResLogger()

        self.transform = None
        self.num_features = None
        self.num_classes = None

    def _get_properties(self, dataset: Dataset, split_idx: int = 0) -> None:
        r"""Avoid triggering transform when getting simple properties."""

        """See `pyg.data.dataset.num_node_features`"""
        data = dataset.get(dataset.indices()[split_idx])
        # Do not fill cache for `InMemoryDataset`:
        if hasattr(dataset, '_data_list') and dataset._data_list is not None:
            dataset._data_list[0] = None
        data = data[0] if isinstance(data, tuple) else data
        if hasattr(data, 'num_node_features'):
            self.num_features = data.num_node_features
        else:
            raise AttributeError(f"'{data.__class__.__name__}' object has no "
                                 f"attribute 'num_node_features'")

        """See `pyg.data.dataset.num_classes`"""
        data_list = [dataset.get(i) for i in dataset.indices()]
        data_list = _get_flattened_data_list(data_list)
        if 'y' in data_list[0] and isinstance(data_list[0].y, torch.Tensor):
            y = torch.cat([data.y for data in data_list if 'y' in data], dim=0)
        else:
            y = torch.as_tensor([data.y for data in data_list if 'y' in data])

        # Do not fill cache for `InMemoryDataset`:
        if hasattr(dataset, '_data_list') and dataset._data_list is not None:
            dataset._data_list = dataset.len() * [None]
        self.num_classes = dataset._infer_num_classes(y)

    def _T_append(self, new_t: List[T.BaseTransform]) -> T.Compose:
        if self.transform is None:
            self.transform = T.Compose(new_t)
        elif isinstance(self.transform, T.Compose):
            self.transform.transforms.extend(new_t)
        elif isinstance(self.transform, T.BaseTransform):
            self.transform = T.Compose([self.transform] + new_t)
        else:
            raise TypeError(f"Invalid transform type: {type(self.transform)}")
        return self.transform

    def get(self, args: Namespace) -> Dataset:
        r"""Load dataset based on parameters.

        Args:
            args.normg (float): Generalized graph norm.

        Returns (update in args):
            args.multi (bool): True for multi-label classification.
            args.num_features (int): Number of input features.
            args.num_classes (int): Number of output classes.
        """
        self.logger.debug('-'*20 + f" Loading data: {self} " + '-'*20)

        # Always use [sparse tensor](https://pytorch-geometric.readthedocs.io/en/latest/notes/sparse_tensor.html) instead of edge_index
        assert torch_geometric.typing.WITH_TORCH_SPARSE
        self.transform = T.Compose([
            # T.ToUndirected(),
            T.RemoveIsolatedNodes(),
            T.RemoveDuplicatedEdges(reduce='mean'),
            T.AddRemainingSelfLoops(fill_value=1.0),
            T.NormalizeFeatures(),
            T.ToSparseTensor(remove_edge_index=True),
            Tspec.GenNorm(left=args.normg),
        ])

        if self.data.startswith('ogb'):
            module_name = 'ogb'
            pass
        # Default to load from PyG
        else:
            module_name = 'torch_geometric.datasets'
            if self.data in ['cora', 'citeseer', 'pubmed']:
                class_name = 'Planetoid'
            elif self.data in ["chameleon", "squirrel"]:
                class_name = 'WikipediaNetwork'
            elif self.data in ["cornell", "texas", "wisconsin"]:
                class_name = 'WebKB'
            else:
                raise ValueError(f"Dataset '{self}' not found.")

            kwargs = dict(
                root=DATAPATH,
                name=self.data,
                transform=self.transform,)

        dataset = load_import(class_name, module_name)(**kwargs)
        self._get_properties(dataset, self.split_idx)
        args.num_features, args.num_classes = self.num_features, self.num_classes
        args.multi = False

        # TODO: what is called dataset.data?
        if len(dataset.train_mask.shape) > 1: # multiple splits, use single one.
            dataset.data.train_mask = dataset.train_mask[:, self.split_idx]
            dataset.data.val_mask = dataset.val_mask[:, self.split_idx]
            dataset.data.test_mask = dataset.test_mask[:, self.split_idx]

        self.logger.info(f"[dataset]: {dataset} (features={self.num_features}, classes={self.num_classes})")
        self.res_logger.concat([('data', self.data)])
        return dataset

    def __call__(self, *args, **kwargs):
        return self.get(*args, **kwargs)

    def __str__(self) -> str:
        return self.data
