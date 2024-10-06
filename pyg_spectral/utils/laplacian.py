import torch
from torch import Tensor

from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.utils import add_self_loops, remove_self_loops, scatter, is_torch_sparse_tensor
from torch_geometric.utils.num_nodes import maybe_num_nodes


def get_laplacian(
    edge_index: Adj,
    edge_weight: OptTensor = None,
    normalization: bool | None = None,
    diag: float = 1.0,
    dtype: torch.dtype | None = None,
    num_nodes: int | None = None,
) -> tuple[Tensor, Tensor] | SparseTensor:
    r"""Computes the graph Laplacian of the graph given by :obj:`edge_index`
    and optional :obj:`edge_weight`.
    Remove the normalization of graph adjacency matrix in
    :func:`torch_geometric.utils.get_laplacian`.

    Args:
        edge_index: The edge indices.
        edge_weight: One-dimensional edge weights.
        normalization: The normalization scheme for the graph
            Laplacian:

            1. :obj:`False`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"True"`: Normalization already applied
            :math:`\mathbf{L} = diag * \mathbf{I} - \mathbf{A}`

        diag: Weight of identity when normalization=True.
        dtype: The desired data type of returned tensor
            in case :obj:`edge_weight=None`.
        num_nodes: The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`.
    """
    if isinstance(edge_index, SparseTensor):
        assert edge_weight is None
        edge_index = edge_index.remove_diag()
        edge_index.storage._value = -edge_index.storage._value

        if normalization:
            # L = diag * I - A
            edge_index = edge_index.fill_diag(diag)
        else:
            # L = D - A
            deg = -edge_index.sum(dim=0)
            edge_index = edge_index.set_diag(deg)
        return edge_index

    elif is_torch_sparse_tensor(edge_index):
        import scipy.sparse as sp
        edge_index = edge_index.to_sparse_csr()
        data = edge_index.values().cpu().detach().numpy()
        indices = edge_index.col_indices().cpu().detach().numpy()
        indptr = edge_index.crow_indices().cpu().detach().numpy()
        shape = edge_index.size()
        ei = sp.csr_matrix((-data, indices, indptr), shape=list(shape))
        edtype = edge_index.values().dtype
        device = edge_index.device

        ei.setdiag(0)
        if normalization:
            # L = diag * I - A
            ei.setdiag(diag)
        else:
            # L = D - A
            deg = -ei.sum(axis=0)
            ei.setdiag(deg)
        ei.eliminate_zeros()

        return torch.sparse_csr_tensor(
            crow_indices=torch.tensor(ei.indptr, dtype=torch.long),
            col_indices=torch.tensor(ei.indices, dtype=torch.long),
            values=torch.tensor(ei.data, dtype=edtype),
            size=shape, device=device)

    else:
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), dtype=dtype,
                                    device=edge_index.device)

        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        row, col = edge_index[0], edge_index[1]
        deg = scatter(edge_weight, row, 0, dim_size=num_nodes, reduce='sum')

        if normalization:
            # L = diag * I - A
            assert isinstance(edge_weight, Tensor)
            edge_index, edge_weight = add_self_loops(  #
                edge_index, -edge_weight, fill_value=diag, num_nodes=num_nodes)
        else:
            # L = D - A
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            edge_weight = torch.cat([-edge_weight, deg], dim=0)

        return edge_index, edge_weight
