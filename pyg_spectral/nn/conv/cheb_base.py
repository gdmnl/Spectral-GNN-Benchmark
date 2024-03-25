from typing import Final

import torch
from torch import Tensor

from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import spmm, add_self_loops
from torch_geometric.data import Data
import torch_geometric.transforms as T

from pyg_spectral.utils import get_laplacian


class ChebBase(MessagePassing):
    r"""Convolutional layer with Chebyshev Polynomials.
    paper: Convolutional Neural Networks on Graphs with Chebyshev Approximation, Revisited
    ref: https://github.com/ivam-he/ChebNetII/blob/main/main/Chebbase_pro.py

    Args:
        K (int, optional): Number of iterations :math:`K`.
        q (int, optional): The constant for Chebyshev Bases.
    """
    supports_edge_weight: Final[bool] = False
    supports_batch: Final[bool] = False
    supports_decouple: Final[bool] = True
    supports_norm_batch: Final[bool] = False

    def __init__(self, K: int = 0, alpha: float = 0, cached: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(ChebBase, self).__init__(**kwargs)

        self.K = K
        self.alpha = alpha
        self.theta = torch.nn.Parameter(Tensor(self.K+1))
        self.cached = cached
        self._cached_adj_t = None
        self.reset_parameters()

    def reset_parameters(self):
        self.theta.data.fill_(0.0)
        self.theta.data[0]=1.0
        self._cached_adj_t = None

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
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

        Tx_0 = x
        Tx_1 = x  # Dummy.
        out = self.theta[0] * Tx_0

        if self.K > 0:
            Tx_1 = self.propagate(adj_t, x=x)
            out += self.theta[1] * Tx_1

        for i in range(2, self.K+1):
            Tx_2 = self.propagate(adj_t, x=Tx_1)
            Tx_2 = 2*Tx_2 - Tx_0
            out += (self.theta[i] / i**self.alpha) * Tx_2
            Tx_0, Tx_1 = Tx_1, Tx_2

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return f'{self.__class__.__name__}(K={self.K}, alpha={self.alpha})'
