from typing import Any
from argparse import Namespace

import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.utils import index_to_mask
from pyg_spectral.utils import load_import
from ogb.linkproppred import Evaluator


from .utils import resolve_split, resolve_data, T_insert, get_iso_nodes_mapping


CLASS_NAME = 'OGB'
DATA_LIST = ['ogbl-collab']


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
    evaluator = Evaluator(args.data)

    args.multi = False
    args.metric = f"s_{evaluator.eval_metric}"
    args.data_split = f"Original_0"

    kwargs = dict(
        root=datapath.joinpath(CLASS_NAME).resolve().absolute(),
        name=args.data,
        pre_transform=None,
        transform=T_insert(transform, T.ToUndirected(), index=0))

    dataset = load_import('PygLinkPropPredDataset', 'ogb.linkproppred')(**kwargs)
    data = dataset[0]
    del data.edge_weight
    del data.edge_year
    args.in_channels = data.num_node_features
    args.out_channels = 1

    # if args.data == :

    split_edge = dataset.get_edge_split()
    split_name = {'train': 'train',
                  'valid': 'val',
                  'test': 'test'}
    for split in ['train', 'valid', 'test']:
        setattr(data, f"{split_name[split]}_mask", split_edge[split]['edge'].T)
    for split in ['valid', 'test']:
        setattr(data, f"{split_name[split]}_mask_neg", split_edge[split]['edge_neg'].T)
    data.train_mask_neg = torch.Tensor([[0], [1]]).long()

    return data
    # data_test = data.clone()
    # val_edge_index = split_edge['valid']['edge'].t()
    # data_test.edge_index = torch.cat([data.edge_index, val_edge_index], dim=-1)
    # data_test.y = torch.ones(data.edge_index.size(1), dtype=torch.float)
    # return [data, data_test]
