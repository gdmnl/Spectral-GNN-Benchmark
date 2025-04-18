from typing import Any
from argparse import Namespace

from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.utils import index_to_mask
from pyg_spectral.utils import load_import

from .utils import resolve_split, resolve_data, T_insert, get_iso_nodes_mapping


CLASS_NAME = 'OGB'
DATA_LIST = ['ogbn-products', 'ogbn-arxiv', 'ogbn-mag', 'ogbn-papers100M']


class T_ogbn_mag(T.BaseTransform):
    def forward(self, data: Any) -> Any:
        new_data = Data(
            x=data.x_dict['paper'],
            edge_index=data.edge_index_dict[('paper', 'cites', 'paper')],
            y=data.y_dict['paper'],
            num_nodes=data.x_dict['paper'].shape[0])
        return new_data


def get_data(datapath, transform, args: Namespace):
    r"""Load data based on parameters and configurations.

    :paper: Open Graph Benchmark: Datasets for Machine Learning on Graphs.
    :ref: https://ogb.stanford.edu/docs/nodeprop/

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
    args.multi = True if args.data == 'ogbn-proteins' else False
    args.metric = 's_auroc' if args.data == 'ogbn-proteins' else 's_f1i'
    args.data_split = f"Original_0"

    kwargs = dict(
        root=datapath.joinpath(CLASS_NAME).resolve().absolute(),
        name=args.data,
        pre_transform=T_ogbn_mag() if args.data == 'ogbn-mag' else None,
        transform=T_insert(transform, T.ToUndirected(), index=0))

    dataset = load_import('PygNodePropPredDataset', 'ogb.nodeproppred')(**kwargs)
    data = resolve_data(args, dataset)

    idx = dataset.get_idx_split()
    if args.data == 'ogbn-products':
        mapping = get_iso_nodes_mapping(dataset)
        for k in idx:
            idx[k] = mapping[idx[k]]
    elif args.data == 'ogbn-mag':
        for k in idx:
            idx[k] = idx[k]['paper']
    data.train_mask = index_to_mask(idx['train'], data.y.size(0))
    data.val_mask = index_to_mask(idx['valid'], data.y.size(0))
    data.test_mask = index_to_mask(idx['test'], data.y.size(0))
    data = resolve_split(args.data_split, data)

    return data
