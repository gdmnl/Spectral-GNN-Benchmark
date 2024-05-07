# -*- coding:utf-8 -*-
"""
Author: nyLiao
File Created: 2024-02-26
"""
from typing import List, Tuple
from pathlib import Path
from argparse import Namespace
import logging

import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data, Dataset
import torch_geometric.transforms as T
from torch_geometric.data.dataset import _get_flattened_data_list
import pyg_spectral.transforms as Tspec
from pyg_spectral.utils import load_import

from utils import ResLogger


DATAPATH = Path('../data')


def split_random(seed, n, n_train, n_val):
    """Split index randomly"""
    np.random.seed(seed)
    rnd = np.random.permutation(n)

    train_idx = np.sort(rnd[:n_train])
    val_idx = np.sort(rnd[n_train:n_train + n_val])

    train_val_idx = np.concatenate((train_idx, val_idx))
    test_idx = np.sort(np.setdiff1d(np.arange(n), train_val_idx))

    train_mask, val_mask, test_mask = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


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
        if '/' in args.data_split:
            self.data_split = args.data_split
            self.split_idx = 0
        else:
            self.split_idx = int(args.data_split)

        self.logger = logging.getLogger('log')
        self.res_logger = res_logger or ResLogger()

        # Always use [sparse tensor](https://pytorch-geometric.readthedocs.io/en/latest/notes/sparse_tensor.html) instead of edge_index
        assert torch_geometric.typing.WITH_TORCH_SPARSE
        self.transform = T.Compose([
            # T.ToUndirected(),
            T.RemoveIsolatedNodes(),
            T.RemoveDuplicatedEdges(reduce='mean'),
            T.AddRemainingSelfLoops(fill_value=1.0),
            T.NormalizeFeatures(),
            T.ToSparseTensor(remove_edge_index=True),
        ])
        self.num_features = None
        self.num_classes = None

    # ===== Data processing
    def _resolve_split(self, data: Data) -> None:
        if type(self.data_split) is str:
            (r_train, r_val, r_test) = map(int, self.data_split.split('/'))
            n = data.num_nodes
            n_train, n_val = int(np.ceil(n * r_train / 100)), int(np.ceil(n * r_val / 100))

            train_mask, val_mask, test_mask = split_random(self.seed, n, n_train, n_val)
            data.train_mask = torch.as_tensor(train_mask)
            data.val_mask = torch.as_tensor(val_mask)
            data.test_mask = torch.as_tensor(test_mask)
        else:
            if data.train_mask.dim() > 1:
                data.train_mask = data.train_mask[:, self.split_idx]
                data.val_mask = data.val_mask[:, self.split_idx]
                data.test_mask = data.test_mask[:, self.split_idx]
        return data

    def _get_properties(self, dataset: Dataset, data: Data = None) -> None:
        r"""Avoid triggering transform when getting simple properties."""

        """See `pyg.data.dataset.num_node_features`"""
        if data is None:
            data = dataset.get(dataset.indices()[self.split_idx])
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

    # ===== Data acquisition
    def _resolve_import(self, args: Namespace) -> Tuple[str, str, dict]:
        # >>>>>>>>>>
        if self.data.startswith('ogb'):
            module_name = 'ogb'
            pass
        elif self.data in ['chameleon_filtered', 'squirrel_filtered']:
            module_name = 'dataset_process'
            class_name = 'FilteredWikipediaNetwork'
        # Default to load from PyG
        else:
            module_name = 'torch_geometric.datasets'
            # Small-scale: use 60/20/20 split
            if self.data in ['cora', 'citeseer', 'pubmed']:
                class_name = 'Planetoid'
            # elif self.data in ["chameleon", "squirrel"]:
            #     class_name = 'WikipediaNetwork'
            elif self.data in ["cornell", "texas", "wisconsin"]:
                class_name = 'WebKB'
            else:
                raise ValueError(f"Dataset '{self}' not found.")
        # <<<<<<<<<<

            kwargs = dict(
                root=DATAPATH,
                name=self.data,
                transform=self.transform,)
        return module_name, class_name, kwargs

    def get(self, args: Namespace) -> Data:
        r"""Load data based on parameters.

        Args:
            args.normg (float): Generalized graph norm.

        Returns (update in args):
            args.multi (bool): True for multi-label classification.
            args.num_features (int): Number of input features.
            args.num_classes (int): Number of output classes.
        """
        self.logger.debug('-'*20 + f" Loading data: {self} " + '-'*20)

        self._T_append([Tspec.GenNorm(left=args.normg),])
        module_name, class_name, kwargs = self._resolve_import(args)

        dataset = load_import(class_name, module_name)(**kwargs)
        data = dataset[0]
        data = self._resolve_split(data)

        self._get_properties(dataset, data)
        args.num_features, args.num_classes = self.num_features, self.num_classes
        args.multi = False

        self.logger.info(f"[dataset]: {dataset} (features={self.num_features}, classes={self.num_classes})")
        self.res_logger.concat([('data', self.data)])
        del dataset
        return data

    def __call__(self, *args, **kwargs):
        return self.get(*args, **kwargs)

    def __str__(self) -> str:
        return self.data


class SingleGraphLoader_Trial(SingleGraphLoader):
    r"""Reuse necessary data for multiple runs.
    """
    def get(self, args: Namespace) -> Data:
        module_name, class_name, kwargs = self._resolve_import(args)
        dataset = load_import(class_name, module_name)(**kwargs)
        data = dataset[0]
        data = self._resolve_split(data)

        self._get_properties(dataset, data)
        args.num_features, args.num_classes = self.num_features, self.num_classes
        args.multi = False

        self.logger.info(f"[dataset]: {dataset} (features={self.num_features}, classes={self.num_classes})")
        self.res_logger.concat([('data', self.data)])
        return data

    def update(self, args: Namespace, data: Data) -> Data:
        r"""Update data split for the next trial.
        """
        args.num_features, args.num_classes = self.num_features, self.num_classes
        args.multi = False
        return Tspec.GenNorm(left=args.normg)(data)
