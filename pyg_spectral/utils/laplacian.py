from typing import Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.utils import add_self_loops, remove_self_loops, scatter
from torch_geometric.utils.num_nodes import maybe_num_nodes


def get_laplacian(
    edge_index: Adj,
    edge_weight: OptTensor = None,
    normalization: Optional[bool] = None,
    diag: float = 1.0,
    dtype: Optional[torch.dtype] = None,
    num_nodes: Optional[int] = None,
) -> Union[Tuple[Tensor, Tensor], SparseTensor]:
    r"""Computes the graph Laplacian of the graph given by :obj:`edge_index`
    and optional :obj:`edge_weight`.
    Remove the normalization of graph adjacency matrix in
    :class:`torch_geometric.transforms.get_laplacian`.

    Args:
        edge_index (LongTensor or SparseTensor): The edge indices.
        edge_weight (Tensor, optional): One-dimensional edge weights.
            (default: :obj:`None`)
        normalization (bool, optional): The normalization scheme for the graph
            Laplacian (default: :obj:`True`):

            1. :obj:`False`: No normalization
            :math:`\mathbf{L} = \mathbf{D} - \mathbf{A}`

            2. :obj:`"True"`: Normalization already applied
            :math:`\mathbf{L} = diag * \mathbf{I} - \mathbf{A}`

        diag (float, optional): Weight of identity when normalization=True.
            (default: :obj:`1.0`)
        dtype (torch.dtype, optional): The desired data type of returned tensor
            in case :obj:`edge_weight=None`. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
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
            deg = edge_index.sum(dim=0)
            edge_index = edge_index.set_diag(deg)
        return edge_index

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
