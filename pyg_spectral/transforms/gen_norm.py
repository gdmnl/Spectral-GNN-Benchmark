import torch
from torch import Tensor

from torch_geometric.typing import SparseTensor, torch_sparse
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import scatter

import traceback

def pow_with_pinv(x: Tensor, p: float) -> Tensor:
    r"""Inplace power operation :math:`x^p` with pseudo-inverse for :math:`p<0`.
    """
    x = x.pow_(p)
    return x.masked_fill_(x == float('inf'), 0)


@functional_transform('gen_norm')
class GenNorm(BaseTransform):
    r"""Generalized graph normalization from GBP/AGP.

    .. math::
        \mathbf{\hat{A}} = \mathbf{\hat{D}}^{-a} (\mathbf{A} + \mathbf{I})
        \mathbf{\hat{D}}^{-b}

    where :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij} + 1` and
        :math:`a,b \in [0,1]`.

    Args:
        left (float): left (row) normalization :math:`a`.
        right (float): right (col) normalization :math:`b`. Default to :math:`1-a`.
    """
    def __init__(self, left: float, right: float = None,
                 dtype: torch.dtype = torch.float32):
        self.left = left
        self.right = right if right is not None else (1.0 - left)
        self.dtype = dtype

    def forward(self, data: Data) -> Data:
        assert 'edge_index' in data or 'adj_t' in data
        print('='*20)
        traceback.print_stack()
        print('='*20)

        # FIXME: Update to `EdgeIndex` [Release note 2.5.0](https://github.com/pyg-team/pytorch_geometric/releases/tag/2.5.0)
        if 'adj_t' in data and isinstance(data.adj_t, SparseTensor):
            deg_out = torch_sparse.sum(data.adj_t, dim=0)
            deg_out = pow_with_pinv(deg_out, -self.left)
            deg_in = torch_sparse.sum(data.adj_t, dim=1)
            deg_in = pow_with_pinv(deg_in, -self.right)

            data.adj_t = torch_sparse.mul(data.adj_t, deg_in.view(-1, 1))
            data.adj_t = torch_sparse.mul(data.adj_t, deg_out.view(1, -1))
            return data

        elif 'edge_index' in data:
            num_nodes = data.num_nodes
            edge_index = data.edge_index
            if data.edge_weight is None and data.edge_attr is None:
                edge_weight = torch.ones((edge_index.size(1), ),
                    dtype=self.dtype,
                    device=edge_index.device)
                key = 'edge_weight'
            else:
                edge_weight = data.edge_attr if data.edge_weight is None else data.edge_weight
                key = 'edge_attr' if data.edge_weight is None else 'edge_weight'
                assert edge_weight.dim() == 1, "Multi-dimension edge attribute not supported."

            row, col = edge_index

            deg_out = scatter(edge_weight, row, dim=0, dim_size=num_nodes, reduce='sum')
            deg_out = pow_with_pinv(deg_out, -self.left)
            deg_in = scatter(edge_weight, col, dim=0, dim_size=num_nodes, reduce='sum')
            deg_in = pow_with_pinv(deg_in, -self.right)

            edge_weight = deg_out[row] * edge_weight * deg_in[col]
            setattr(data, key, edge_weight)
            return data

        raise NotImplementedError("Only support `edge_index` or `SparseTensor`!")

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'D^(-{self.left}) A D^(-{self.right})')
