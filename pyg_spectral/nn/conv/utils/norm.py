import torch
from torch import Tensor

from torch_geometric.typing import SparseTensor, torch_sparse
from torch_geometric.utils import add_remaining_self_loops, scatter
from torch_geometric.utils.num_nodes import maybe_num_nodes


# TODO: `@torch.jit._overload` [`gcn_norm`](https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/conv/gcn_conv.py)

# TODO: consider migrate to like `torch_geometric.transforms`
def identity_n_norm(edge_index, edge_weight=None, num_nodes=None,
                    rnorm=None, diag=1., dtype=torch.float32):
    r"""Manage normalization and identity of adjacency matrix.

    Args:
        edge_index: Unormalized adjacency matrix
        rnorm (float, optional): Normalization in form $D^{r-1}AD^r$. None for no normalization.
        diag (float, optional): Weight of identity. None for no identity addition.

    Returns:
        if `edge_index` stores float, return `edge_index` with weights
        else return `edge_index` and `edge_weight`
    """
    # TODO: Update to `EdgeIndex` [Release note 2.5.0](https://github.com/pyg-team/pytorch_geometric/releases/tag/2.5.0)
    if isinstance(edge_index, SparseTensor):
        assert edge_index.size(0) == edge_index.size(1)
        if diag is not None:
            edge_index = torch_sparse.fill_diag(edge_index, diag)
        if rnorm is not None:
            # TODO: r-norm
            deg = torch_sparse.sum(edge_index, dim=1)
            deg_inv_sqrt = deg.pow_(-0.5)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
            edge_index = torch_sparse.mul(edge_index, deg_inv_sqrt.view(-1, 1))
            edge_index = torch_sparse.mul(edge_index, deg_inv_sqrt.view(1, -1))
        return edge_index

    if isinstance(edge_index, Tensor):
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        if diag is not None:
            edge_index, edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, diag, num_nodes)
        if rnorm is None:
            if edge_weight is None:
                return edge_index
        else:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
            row, col = edge_index[0], edge_index[1]
            idx = col
            deg = scatter(edge_weight, idx, dim=0, dim_size=num_nodes, reduce='sum')
            deg_inv_sqrt = deg.pow_(-0.5)
            deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
            edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        return edge_index, edge_weight

    raise NotImplementedError()


def to_lap():
    r"""Convert adjacency matrix to Laplacian matrix
    """
    # TODO
    pass
