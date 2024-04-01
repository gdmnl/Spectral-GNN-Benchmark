from typing import Final
import math
import torch
from torch import Tensor

from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import spmm, add_self_loops
from torch_geometric.data import Data
import torch_geometric.transforms as T

from pyg_spectral.utils import get_laplacian


class Adagnn(MessagePassing):
    r"""Convolutional layer with AdaGNN.
    paper: AdaGNN: Graph Neural Networks with Adaptive Frequency Response Filter
    ref: https://github.com/yushundong/AdaGNN/blob/main/layers.py

    Args:
        K (int, optional): Number of iterations :math:`K`.
        q (int, optional): The constant for Chebyshev Bases.
    """
    supports_edge_weight: Final[bool] = False
    supports_batch: Final[bool] = False
    supports_decouple: Final[bool] = True
    supports_norm_batch: Final[bool] = False

    def __init__(self, K: int = 0, in_features: int = 0, hidden_dimension: int = 0, cached: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(Adagnn, self).__init__(**kwargs)
        
        self.K = K
        assert K - 2 >= 0
        self.weight = torch.nn.Parameter(torch.FloatTensor(in_features, hidden_dimension)) # the first layer
        self.learnable_diag_1 = torch.nn.Parameter(torch.FloatTensor(in_features))  
        self.theta = [torch.nn.Parameter(torch.FloatTensor(hidden_dimension)) for i in range(K-1)]
 
        self.cached = cached
        self._cached_adj_t = None
        self.reset_parameters()
 
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        torch.nn.init.normal_(self.learnable_diag_1, mean=0, std=0)
        for i in range(len(self.theta)):
            torch.nn.init.normal_(self.theta[i], mean=0, std=0)

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
       
        alpha = torch.diag(self.learnable_diag_1)
        e2 = torch.mm(x, alpha)
        # print("alpha", alpha.requires_grad)
        e1 = self.propagate(adj_t, x=e2)
        e4 = torch.sub(x, e1)
        Tx_0 = torch.mm(e4, self.weight)

        for i in range(len(self.theta)):
            alpha = torch.diag(self.theta[i]).cuda()
            e2 = torch.mm(Tx_0, alpha)
            # print("alpha", alpha.requires_grad)
            e1 = self.propagate(adj_t, x=e2)
            e4 = torch.sub(Tx_0, e1)
            Tx_0 = e4

        out = Tx_0
        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return f'{self.__class__.__name__}(K={self.K})'