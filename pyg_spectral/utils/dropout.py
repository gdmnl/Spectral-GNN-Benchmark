from typing import Tuple

import torch
from torch import Tensor

from torch_geometric.typing import SparseTensor
import torch_geometric.utils as pyg_utils


def dropout_edge(edge_index: Tensor, p: float = 0.5,
                 force_undirected: bool = False,
                 training: bool = True) -> Tuple[Tensor, Tensor]:
    r"""Random inplace edge dropout for the adjacency matrix
    :obj:`edge_index` with probability :obj:`p` using samples from
    a Bernoulli distribution.
    Expand :func:`torch_geometric.utils.dropout_edge` with type support.

    Args:
        edge_index: The edge indices.
        p: Dropout probability.
        force_undirected: If set to :obj:`True`, will either
            drop or keep both edges of an undirected edge.
        training: If set to :obj:`False`, this operation is a no-op.

    Returns:
        edge_index, edge_mask (LongTensor, BoolTensor): The edge indices and the edge mask.
    """
    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 (got {p})')

    if isinstance(edge_index, SparseTensor):
        if not training or p == 0.0:
            edge_mask = edge_index.new_ones(edge_index.sparse_size()[0], dtype=torch.bool)
            return edge_index, edge_mask

        edge_tensor, _ = pyg_utils.to_edge_index(edge_index)
        _, edge_mask = pyg_utils.dropout_edge(edge_tensor, p, force_undirected, training)

        return edge_index.masked_select_nnz(edge_mask), edge_mask

    elif pyg_utils.is_torch_sparse_tensor(edge_index):
        if not training or p == 0.0:
            edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
            return edge_index, edge_mask

        edge_tensor, _ = pyg_utils.to_edge_index(edge_index)
        sparse_mask, edge_mask = pyg_utils.dropout_edge(edge_tensor, p, force_undirected, training)

        sparse_mask = pyg_utils.to_torch_coo_tensor(sparse_mask, edge_mask, edge_index.size(1), is_coalesced=True)
        return edge_index.sparse_mask(sparse_mask), edge_mask

    else:
        return pyg_utils.dropout_edge(edge_index, p, force_undirected, training)
