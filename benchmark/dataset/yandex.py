import os.path as osp
from typing import Callable, Optional
from argparse import Namespace

import numpy as np
import torch

from torch_geometric.data import Data, InMemoryDataset, download_url
from torch_geometric.utils import coalesce
import torch_geometric.transforms as T

from .utils import resolve_split, resolve_data


CLASS_NAME = 'Yandex'
DATA_LIST = ['chameleon_filtered', 'squirrel_filtered', \
             'roman_empire', 'amazon_ratings', 'minesweeper', 'tolokers', 'questions']


class Yandex(InMemoryDataset):
    r"""
    :paper: A critical look at the evaluation of GNNs under heterophily: are we really making progress?
    :ref: https://github.com/yandex-research/heterophilous-graphs
    """
    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.name = name.lower()
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        return f'{self.name}.npz'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        url = "https://github.com/yandex-research/heterophilous-graphs/raw/main/data"
        download_url(f'{url}/{self.name}.npz', self.raw_dir)

    def process(self) -> None:
        data = np.load(osp.join(self.raw_dir, f'{self.name}.npz'), allow_pickle=True)
        x = torch.tensor(data['node_features'], dtype=torch.float)
        y = torch.tensor(data['node_labels'], dtype=torch.long)
        edge_index = torch.tensor(data['edges'], dtype=torch.long).t().contiguous()
        edge_index = coalesce(edge_index, num_nodes=x.size(0))

        train_masks = torch.tensor(data['train_masks'], dtype=torch.bool)
        val_masks = torch.tensor(data['val_masks'], dtype=torch.bool)
        test_masks = torch.tensor(data['test_masks'], dtype=torch.bool)

        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_masks,
                    val_mask=val_masks, test_mask=test_masks)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])


def get_data(datapath, transform, args: Namespace):
    r"""Load data based on parameters and configurations.

    Args:
        datapath: Path to the root data directory.
        transform: Data transformation pipeline.
        args: Parameters.

            * args.data (str): Dataset name.
            * args.data_split (str): Index of dataset split.
    Returns:
        data (Data): The resolved data sample from the dataset.
    Updates:
        args.in_channels (int): Number of input features.
        args.out_channels (int): Number of output classes.
        args.multi (bool): True for multi-label classification.
        args.metric (str): Main metric name for evaluation.
    """
    args.multi = False
    args.metric = {
        'chameleon_filtered': 's_f1i',
        'squirrel_filtered': 's_f1i',
        'roman_empire': 's_f1i',
        'amazon_ratings': 's_f1i',
        'minesweeper': 's_auroc',
        'tolokers': 's_auroc',
        'questions': 's_auroc'
        }[args.data]
    assert args.data_split.split('_')[0] in ['Random', 'Stratify']

    kwargs = dict(
        root=datapath.joinpath(CLASS_NAME).resolve().absolute(),
        name=args.data,
        pre_transform=T.ToUndirected(),
        transform=transform)

    dataset = Yandex(**kwargs)
    data = resolve_data(args, dataset)
    data = resolve_split(args.data_split, data)

    return data
