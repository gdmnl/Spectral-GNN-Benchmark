# -*- coding:utf-8 -*-
"""
Author: nyLiao
File Created: 2024-02-26
"""
from typing import List, Tuple
from pathlib import Path
from argparse import Namespace
import logging

import torch
import torch_geometric
from torch_geometric.data import Data, Dataset
import torch_geometric.transforms as T
from torch_geometric.data.dataset import _get_flattened_data_list

import pyg_spectral.transforms as Tspec
from pyg_spectral.utils import load_import

from utils import ResLogger
from dataset_process import idx2mask, split_random
from dataset_process.linkx import T_arxiv_year


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
    def _resolve_split(self, dataset: Dataset, data: Data) -> None:
        if hasattr(self, 'data_split'):
            (r_train, r_val) = map(int, self.data_split.split('/')[:2])
            r_train, r_val = r_train / 100, r_val / 100

            train_mask, val_mask, test_mask = split_random(data.y, r_train, r_val)
            data.train_mask = torch.as_tensor(train_mask)
            data.val_mask = torch.as_tensor(val_mask)
            data.test_mask = torch.as_tensor(test_mask)
        else:
            if self.data.startswith('ogbn-'):
                data.train_mask, data.val_mask, data.test_mask = idx2mask(dataset.get_idx_split(), data.y.size(0))
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

    def _T_prepend(self, new_t: List[T.BaseTransform]) -> T.Compose:
        if self.transform is None:
            self.transform = T.Compose(new_t)
        elif isinstance(self.transform, T.Compose):
            self.transform.transforms = new_t + self.transform.transforms
        elif isinstance(self.transform, T.BaseTransform):
            self.transform = T.Compose(new_t + [self.transform])
        else:
            raise TypeError(f"Invalid transform type: {type(self.transform)}")
        return self.transform

    # ===== Data acquisition
    def _resolve_import(self, args: Namespace) -> Tuple[str, str, dict]:
        # >>>>>>>>>>
        if self.data.startswith('ogbn-'):
            module_name = 'ogb.nodeproppred'
            class_name = 'PygNodePropPredDataset'
            kwargs = dict(
                root=DATAPATH.joinpath('OGB'),
                name=self.data,
                transform=self._T_prepend(T.ToUndirected()),)
        elif self.data in ['arxiv-year']:
            module_name = 'ogb.nodeproppred'
            class_name = 'PygNodePropPredDataset'
            kwargs = dict(
                root=DATAPATH.joinpath('LINKX'),
                name='ogbn-arxiv',
                pre_transform=T_arxiv_year(),
                transform=self.transform,)
        # FIXME: check ToUndirected in LINKX
        elif self.data in ['genius', 'pokec', 'snap-patents', 'twitch-gamer', 'wiki']:
            module_name = 'dataset_process'
            class_name = 'LINKX'
            kwargs = dict(
                root=DATAPATH.joinpath('LINKX'),
                name=self.data,
                transform=self.transform,)
        elif self.data in ['penn94', 'amherst41', 'cornell5', 'johns_hopkins55', 'reed98']:
            module_name = 'dataset_process'
            class_name = 'FB100'
            kwargs = dict(
                root=DATAPATH.joinpath('LINKX'),
                name=self.data,
                transform=self.transform,)
        elif self.data in ['chameleon_filtered', 'squirrel_filtered', \
                'roman_empire', 'amazon_ratings', 'minesweeper', 'tolokers', 'questions']:
            module_name = 'dataset_process'
            class_name = 'Yandex'
            kwargs = dict(
                root=DATAPATH.joinpath('Yandex'),
                name=self.data,
                pre_transform=T.ToUndirected(),
                transform=self.transform)
        # Default to load from PyG
        else:
            module_name = 'torch_geometric.datasets'
            kwargs = dict(
                root=DATAPATH.joinpath('PyG'),
                name=self.data,
                transform=self.transform,)
            pyg_mapping = {
                'cora':         'Planetoid',
                'citeseer':     'Planetoid',
                'pubmed':       'Planetoid',
                'photo':        'Amazon',
                'computers':    'Amazon',
                'cs':           'Coauthor',
                'physics':      'Coauthor',
                'ego-facebook': 'SNAPDataset',
                'ego-twitter':  'SNAPDataset',
                'ego-gplus':    'SNAPDataset',
                'facebook':     'AttributedGraphDataset',
                'tweibo':       'AttributedGraphDataset',
                # 'chameleon':    'WikipediaNetwork',
                # 'squirrel':     'WikipediaNetwork',
                'cornell':      'WebKB',
                'texas':        'WebKB',
                'wisconsin':    'WebKB',
            }
            if self.data in pyg_mapping:
                class_name = pyg_mapping[self.data]
            elif self.data in ["flickr", "reddit"]:
                class_name = self.data.capitalize()
                kwargs.pop('name')
            else:
                raise ValueError(f"Dataset '{self}' not found.")
        # <<<<<<<<<<

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
        data = self._resolve_split(dataset, data)

        self._get_properties(dataset, data)
        args.num_features, args.num_classes = self.num_features, self.num_classes
        args.multi = False

        # Remaining resolvers
        if not args.multi and data.y.dim() > 1 and data.y.size(1) == 1:
            data.y = data.y.flatten()

        self.logger.info(f"[dataset]: {dataset} (features={self.num_features}, classes={self.num_classes})")
        self.logger.info(f"[data]: {data}")
        split_dict = {k[:-5]: v.sum().item() for k, v in data.items() if k.endswith('_mask')}
        self.logger.info(f"[split]: {split_dict}")
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
        data = self._resolve_split(dataset, data)

        self._get_properties(dataset, data)
        args.num_features, args.num_classes = self.num_features, self.num_classes
        args.multi = False

        self.res_logger.concat([('data', self.data)])
        return data

    def update(self, args: Namespace, data: Data) -> Data:
        r"""Update data split for the next trial.
        """
        args.num_features, args.num_classes = self.num_features, self.num_classes
        args.multi = False
        return Tspec.GenNorm(left=args.normg)(data)
