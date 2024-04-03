from typing import Final

from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor, PairTensor
import math
import torch
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, remove_self_loops #,get_laplacian
from pyg_spectral.utils import get_laplacian
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch.nn.functional as F
from scipy.special import comb

class BernConv(MessagePassing):

    r"""Convolutional layer with Bernstein Polynomials.

    Args:
        K (int, optional): Number of iterations :math:`K`.
    """

    supports_edge_weight: Final[bool] = False
    supports_batch: Final[bool] = False
    supports_decouple: Final[bool] = True
    supports_norm_batch: Final[bool] = False

    def __init__(self, K: int = 0, cached: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(BernConv, self).__init__(**kwargs)

        self.K = K
        self.theta = Parameter(torch.Tensor(self.K+1))
        self.cached = cached
        self._cached_adj_t = None
        self.reset_parameters()
    
    def reset_parameters(self):
        self.theta.data.fill_(0.0)
        self.theta.data[0]=1.0
        self._cached_adj_t = None

    def forward(self,
                x: Tensor,
                edge_index: Adj,
                edge_weight: OptTensor=None,
        ) -> Tensor:

        cache = self._cached_adj_t
        if cache is None:
            if isinstance(edge_index, SparseTensor):
                # A_norm -> L_norm
                edge_index = get_laplacian(
                    edge_index,
                    normalization=True,
                    dtype=x.dtype)
                # L_norm -> L_norm - I
                diag = edge_index.get_diag()
                edge_index.set_diag(diag - 1.0)
                adj_t = edge_index

            else:
                # A_norm -> L_norm
                edge_index, edge_weight = get_laplacian(
                    edge_index, edge_weight,
                    normalization=True,
                    dtype=x.dtype,
                    num_nodes=x.size(self.node_dim))
                # L_norm -> L_norm - I
                edge_index, edge_weight = add_self_loops(
                    edge_index, edge_weight,
                    fill_value=-1.0,
                    num_nodes=x.size(self.node_dim))
                data = Data(edge_index=edge_index, edge_attr=edge_weight)
                data.num_nodes = x.size(self.node_dim)
                adj_t = T.ToSparseTensor(remove_edge_index=True)(data).adj_t

            if self.cached:
                self._cached_adj_t = adj_t
        else:
            adj_t = cache

        tmp = [x]
        for i in range(self.K):
                x = self.propagate(adj_t,x=x)
                tmp.append(x)

        out = (comb(self.K,0)/(2**self.K))*self.theta[0]*tmp[self.K]

        for i in range(self.K):
                x = tmp[self.K-i-1]
                x = self.propagate(adj_t,x=x)
                for j in range(i):
                        x = self.propagate(adj_t,x=x)

                out += (comb(self.K,i+1)/(2**self.K)) * self.theta[i+1] * x

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}(K={})'.format(
            self.__class__.__name__, self.K)