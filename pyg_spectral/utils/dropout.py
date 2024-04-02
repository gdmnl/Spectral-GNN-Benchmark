from typing import Tuple

import torch
from torch import Tensor

from torch_geometric.typing import SparseTensor
from torch_geometric.utils import dropout_edge as dropout_edge_pyg


def dropout_edge(edge_index: Tensor, p: float = 0.5,
                 force_undirected: bool = False,
                 training: bool = True) -> Tuple[Tensor, Tensor]:
    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    if isinstance(edge_index, SparseTensor):
        if not training or p == 0.0:
            edge_mask = edge_index.new_ones(edge_index.sparse_size()[0], dtype=torch.bool)
            return edge_index, edge_mask

        row, col, value = edge_index.coo()
        edge_tensor = torch.stack([row, col], dim=0)

        _, edge_mask = dropout_edge_pyg(edge_tensor, p, force_undirected, training)

        return edge_index.masked_select_nnz(edge_mask), edge_mask

    else:
        return dropout_edge_pyg(edge_index, p, force_undirected, training)
