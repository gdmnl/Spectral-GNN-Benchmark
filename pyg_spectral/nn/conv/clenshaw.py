from typing import Final

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import spmm, add_self_loops
from torch_geometric.data import Data
import torch_geometric.transforms as T

from pyg_spectral.utils import get_laplacian


class ClenShaw(MessagePassing):
    r"""Convolutional layer with ClenShaw GCN.
    paper: Clenshaw Graph Neural Networks
    ref: https://github.com/yuziGuo/ClenshawGNN/blob/master/models/ChebClenshawNN.py

    Args:
        K (int, optional): Number of iterations :math:`K`.
        lambda (float, optional): Eigenvalue.
    """
    supports_edge_weight: Final[bool] = False
    supports_batch: Final[bool] = False
    supports_decouple: Final[bool] = True
    supports_norm_batch: Final[bool] = False

    def __init__(self, K: int = 0, lamda: float=0, cached: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(ChebBase, self).__init__(**kwargs)

        self.K = K

        self.lamda = lamda
        self.thetas = th.log(lamda / (th.arange(n_layers)+1) + 1)
        _ones = th.ones_like(self.thetas)
        self.thetas = th.where(self.thetas<1, self.thetas, _ones)

        self.alphas = Parameter(torch.Tensor(self.K+1))
        self.cached = cached
        self._cached_adj_t = None
        self.reset_parameters()

    def reset_parameters(self):
        self.alphas.data.fill_(0.0)
        self.alphas.data[0]=1.0
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
        Tx_last = torch.zeros_like(Tx_0)
        Tx_last_2 = torch.zeros_like(Tx_0)

        for i in range(1, self.K+1):
            x = self.propagate(edge_index_tilde,x=Tx_last,norm=norm_tilde,size=None)
            x = self.alphas[-(i)] * Tx_0 + 2*x - Tx_last_2
            # FIX ME: original clenshaw gnn has linear weight in each hop.
            Tx_last_2 = Tx_last
            Tx_last = x

        return x

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return f'{self.__class__.__name__}(K={self.K}, alpha={self.lamda})'
