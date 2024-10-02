from typing import Tuple
from argparse import Namespace
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch_geometric.data import Data, Dataset
import torch_geometric.transforms as T
from torch_geometric.utils import index_to_mask


def T_insert(transform, new_t: T.BaseTransform, index=-1) -> T.Compose:
    if transform is None:
        transform = T.Compose([new_t])
    elif isinstance(transform, T.Compose):
        index = len(transform.transforms) + 1 + index if index < 0 else index
        transform.transforms.insert(index, new_t)
    elif isinstance(transform, T.BaseTransform):
        transform = T.Compose([transform].insert(index, new_t))
    else:
        raise TypeError(f"Invalid transform type: {type(transform)}")
    return transform


def resolve_data(args: Namespace, dataset: Dataset) -> Data:
    r"""Acquire data and properties from dataset.

    Args:
        args: Parameters.

            * args.multi (bool): ``True`` for multi-label classification.
        dataset: PyG dataset object.
    Returns:
        data (Data): The resolved PyG data object from the dataset.
    Updates:
        args.num_features (int): Number of input features.
        args.num_classes (int): Number of output classes.
    """
    # Avoid triggering transform when getting simple properties.
    # data = dataset.get(dataset.indices()[0])
    # if hasattr(dataset, '_data_list') and dataset._data_list is not None:
    #     dataset._data_list[0] = None
    # data = data[0] if isinstance(data, tuple) else data
    data = dataset[0]

    assert hasattr(data, 'num_node_features')
    args.num_features = data.num_node_features
    args.num_classes = dataset._infer_num_classes(data.y)

    # Remaining resolvers
    if not args.multi and data.y.dim() > 1 and data.y.size(1) == 1:
        data.y = data.y.flatten()
    # if not args.multi and self.num_classes == 2:
    #     args.num_classes = self.num_classes = 1
    #     data.y = data.y.unsqueeze(1).float()
    return data


def resolve_split(data_split: str, data: Data) -> Data:
    r"""Apply data split masks.

    Args:
        data_split: Index of dataset split, formatted as ``scheme_split`` or ``scheme_split_seed``.

            * ``scheme='Random'``: Random split, ``split`` is ``train/val/test`` ratio.
            * ``scheme='Stratify'``: Stratified split, ``split`` is ``train/val/test`` ratio.
            * ``scheme='Original'``: Original split, ``split`` is the index of split.
        data: PyG data object containing the dataset and its attributes.
    Returns:
        data (Data): The updated PyG data object with split masks (train/val/test).
    """
    ctx = data_split.split('_')
    if len(ctx) == 2:
        scheme, split = ctx
        seed = None
    else:
        scheme, split, seed = ctx
    scheme = scheme.capitalize()

    if scheme in ['Random', 'Stratify']:
        (r_train, r_val) = map(int, split.split('/')[:2])
        r_train, r_val = r_train / 100, r_val / 100

        assert data.num_nodes == data.y.shape[0]
        data.train_mask, data.val_mask, data.test_mask = split_crossval(
            data.y, r_train, r_val,
            seed=int(seed),
            ignore_neg=True,
            stratify=(scheme == 'Stratify'))
    else:
        assert hasattr(data, 'train_mask') and hasattr(data, 'val_mask') and hasattr(data, 'test_mask')
        if data.train_mask.dim() > 1:
            split = int(split) % data.train_mask.size(1)
            data.train_mask = data.train_mask[:, split]
            data.val_mask = data.val_mask[:, split]
            data.test_mask = data.test_mask[:, split]

    data.train_mask = torch.as_tensor(data.train_mask)
    data.val_mask = torch.as_tensor(data.val_mask)
    data.test_mask = torch.as_tensor(data.test_mask)
    return data


def split_crossval(label: torch.Tensor,
                   r_train: float,
                   r_val: float,
                   seed: int = None,
                   ignore_neg: bool =True,
                   stratify: bool =False) -> Tuple[torch.Tensor]:
    r"""Split index by cross-validation"""
    node_labeled = torch.where(label >= 0)[0] if ignore_neg else np.arange(label.shape[0])

    train_idx, val_idx = train_test_split(node_labeled,
                            test_size=r_val,
                            train_size=r_train,
                            random_state=seed,
                            stratify=label[node_labeled] if stratify else None)
    used_idx = np.concatenate((train_idx, val_idx))
    test_idx = np.setdiff1d(node_labeled, used_idx, assume_unique=True)

    return (index_to_mask(torch.as_tensor(train_idx), size=label.shape[0]),
            index_to_mask(torch.as_tensor(val_idx), size=label.shape[0]),
            index_to_mask(torch.as_tensor(test_idx), size=label.shape[0]))


def even_quantile_labels(vals: np.ndarray, nclasses: int, verbose:bool=True):
    """ partitions vals into nclasses by a quantile based split,
    where the first class is less than the 1/nclasses quantile,
    second class is less than the 2/nclasses quantile, and so on

    Args:
        vals: The input array to be partitioned.
        nclasses: The number of classes to partition the array into.
        verbose: Prints the intervals for each class.
    """
    label = -1 * np.ones(vals.shape[0], dtype=int)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.nanquantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        for class_idx, interval in enumerate(interval_lst):
            print(f'Class {class_idx}: [{interval[0]}, {interval[1]})]')
    return label


def get_iso_nodes_mapping(dataset):
    data = dataset.get(dataset.indices()[0])
    edge_index = data.edge_index
    src, dst = edge_index[0], edge_index[1]
    bin = torch.zeros(data.num_nodes, dtype=torch.bool)
    bin[src] = True
    bin[dst] = True
    kept_nodes = torch.where(bin)[0]
    mapping = torch.zeros(data.num_nodes, dtype=torch.long) - 1
    mapping[kept_nodes] = torch.arange(kept_nodes.shape[0])
    return mapping
