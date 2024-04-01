from typing import Final

import math
import torch
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import spmm, add_self_loops
from torch_geometric.data import Data
import torch_geometric.transforms as T
import torch.nn.functional as F

from pyg_spectral.utils import get_laplacian

def cheby(i,x):
    if i==0:
        return 1
    elif i==1:
        return x
    else:
        T0=1
        T1=x
        for ii in range(2,i+1):
            T2=2*x*T1-T0
            T0,T1=T1,T2
        return T2


class ChebConv2(MessagePassing):

    r"""Convolutional layer with Chebyshev II Polynomials.

    Args:
        K (int, optional): Number of iterations :math:`K`.
        Init (bool, optional): If inialize the coefficients for Chebyshev Bases.
    """

    supports_edge_weight: Final[bool] = False
    supports_batch: Final[bool] = False
    supports_decouple: Final[bool] = True
    supports_norm_batch: Final[bool] = False

    def __init__(self, K: int = 0, Init: bool = False, cached: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(ChebConv2, self).__init__(**kwargs)

        self.K = K
        self.Init=Init
        self.theta = Parameter(torch.Tensor(self.K+1))
        self.cached = cached
        self._cached_adj_t = None
        self.reset_parameters()

    def reset_parameters(self):
        self.theta.data.fill_(1.0)
        if self.Init:
            for j in range(self.K+1):
                x_j=math.cos((self.K-j+0.5)*math.pi/(self.K+1))
                self.theta.data[j] = x_j**2
        self._cached_adj_t = None

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
    ) -> Tensor:

        coe_tmp=F.relu(self.theta)
        coe=coe_tmp.clone()

        for i in range(self.K+1):
            coe[i]=coe_tmp[0]*cheby(i,math.cos((self.K+0.5)*math.pi/(self.K+1)))
            for j in range(1,self.K+1):
                x_j=math.cos((self.K-j+0.5)*math.pi/(self.K+1))
                coe[i]=coe[i]+coe_tmp[j]*cheby(i,x_j)
            coe[i]=2*coe[i]/(self.K+1)

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
        out = coe[0]/2*Tx_0

        if self.K > 0:
            Tx_1 = self.propagate(adj_t, x=x)
            out += coe[1] * Tx_1

        for i in range(2, self.K+1):
            Tx_2 = self.propagate(adj_t, x=Tx_1)
            Tx_2 = 2*Tx_2 - Tx_0
            out += coe[i] * Tx_2
            Tx_0, Tx_1 = Tx_1, Tx_2

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={})'.format(self.__class__.__name__, self.K)