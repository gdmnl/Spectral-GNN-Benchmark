from argparse import Namespace

import torch_geometric.transforms as T
from pyg_spectral.utils import load_import

from .utils import resolve_split, resolve_data, T_insert


CLASS_NAME = 'PyG'
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
DATA_LIST = ["flickr", "reddit", "actor"] + list(pyg_mapping.keys())


def get_data(datapath, transform, args: Namespace):
    r"""Load data based on parameters and configurations.

    :paper:  Fast Graph Representation Learning with PyTorch Geometric
    :ref: https://pytorch-geometric.readthedocs.io/en/stable/modules/datasets.html

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
    args.metric = 's_f1i'
    assert args.data_split.split('_')[0] in ['Random', 'Stratify']

    kwargs = dict(
        root=datapath.joinpath(CLASS_NAME),
        name=args.data,
        transform=T_insert(transform, T.ToUndirected(), index=0))
    if args.data in pyg_mapping:
        class_name = pyg_mapping[args.data]
    else:
        class_name = args.data.capitalize()
        kwargs['root'] = kwargs['root'].joinpath(class_name)
        kwargs.pop('name')
    kwargs['root'] = kwargs['root'].resolve().absolute()

    dataset = load_import(class_name, 'torch_geometric.datasets')(**kwargs)
    data = resolve_data(args, dataset)
    data = resolve_split(args.data_split, data)

    return data
