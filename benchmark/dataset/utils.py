from argparse import Namespace
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.data.dataset import _get_flattened_data_list
import torch_geometric.transforms as T


def idx2mask(idx: dict, n: int) -> tuple:
    res = tuple()
    for k in ['train', 'valid', 'test']:
        mask = torch.zeros(n, dtype=bool)
        mask[idx[k]] = True
        res += (mask,)
    return res


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


def get_split(data_split: str, data: Data) -> Data:
    # TODO: support more split schemes
    scheme, split = data_split.split('_')
    if scheme == 'Random':
        (r_train, r_val) = map(int, split.split('/')[:2])
        r_train, r_val = r_train / 100, r_val / 100

        data.train_mask, data.val_mask, data.test_mask = split_random(data.y, r_train, r_val)
    else:
        assert hasattr(data, 'train_mask') and hasattr(data, 'val_mask') and hasattr(data, 'test_mask')
        if data.train_mask.dim() > 1:
            split = int(split)
            if split >= data.train_mask.size(1):
                split = split % data.train_mask.size(1)
            data.train_mask = data.train_mask[:, split]
            data.val_mask = data.val_mask[:, split]
            data.test_mask = data.test_mask[:, split]

    data.train_mask = torch.as_tensor(data.train_mask)
    data.val_mask = torch.as_tensor(data.val_mask)
    data.test_mask = torch.as_tensor(data.test_mask)
    return data


def split_random(label: torch.Tensor, r_train: float, r_val: float, ignore_neg=True) -> tuple:
    """Split index randomly"""
    node_labeled = torch.where(label >= 0)[0] if ignore_neg else label
    n = node_labeled.shape[0]
    n_train, n_val = int(np.ceil(n * r_train)), int(np.ceil(n * r_val))
    rnd = np.random.permutation(n)

    train_idx = np.sort(rnd[:n_train])
    val_idx = np.sort(rnd[n_train:n_train + n_val])
    train_val_idx = np.concatenate((train_idx, val_idx))
    test_idx = np.sort(np.setdiff1d(np.arange(n), train_val_idx))

    if ignore_neg:
        idx = {'train': node_labeled[train_idx],
               'valid': node_labeled[val_idx],
               'test': node_labeled[test_idx]}
        return idx2mask(idx, label.shape[0])
    else:
        idx = {'train': train_idx, 'valid': val_idx, 'test': test_idx}
        return idx2mask(idx, n)


def even_quantile_labels(vals, nclasses, verbose=True):
    """ partitions vals into nclasses by a quantile based split,
    where the first class is less than the 1/nclasses quantile,
    second class is less than the 2/nclasses quantile, and so on

    vals is np array
    returns an np array of int class labels
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
        print('Class Label Intervals:')
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
