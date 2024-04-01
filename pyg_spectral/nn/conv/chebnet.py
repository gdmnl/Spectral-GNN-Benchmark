import math
import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops #,get_laplacian
from pyg_spectral.utils import get_laplacian
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import spmm, dropout_edge


""" not supporting edge_drop yet"""

class ChebConv(MessagePassing):
    r"""Convolutional layer with Chebyshev Polynomials.

    Args:
        K (int, optional): Number of iterations :math:`K`.
        q (int, optional): The constant for Chebyshev Bases.
    """

    def __init__(self, K, dropedge, **kwargs):
        super(ChebConv, self).__init__(aggr='add', **kwargs)
        
        self.K = K
        self.scheme = 'custom'
        self.temp = Parameter(torch.Tensor(self.K+1))
        self.q=0 #The positive constant
        self.dropedge = dropedge
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(0.0)
        self.temp.data[0]=1.0

    def forward(self, x, edge_index, edge_weight=None):
        coe=self.temp
        #L
        #original normalization is: edge_index1, norm1 = get_laplacian(edge_index, edge_weight, normalization='sym', dtype=x.dtype, num_nodes=x.size(self.node_dim))
        edge_index1, norm1 = get_laplacian(edge_index, edge_weight, normalization=True, dtype=x.dtype, num_nodes=x.size(self.node_dim))

        #L_tilde=L-I
        edge_index_tilde, norm_tilde= add_self_loops(edge_index1, norm1, fill_value=-1.0, num_nodes=x.size(self.node_dim))

        Tx_0=x
        Tx_1=self.propagate(edge_index_tilde,x=x,norm=norm_tilde,size=None)

        out=coe[0]*Tx_0+coe[1]*Tx_1

        for i in range(2,self.K+1):
            Tx_2=self.propagate(edge_index_tilde,x=Tx_1,norm=norm_tilde,size=None)
            Tx_2=2*Tx_2-Tx_0
            out=out+(coe[i]/i**self.q)*Tx_2
            Tx_0,Tx_1 = Tx_1, Tx_2
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)


class ChebNet(ChebConv):

    def __init__(self, in_channels: int, out_channels: int, K: int = 0, dropedge: float = 0., bias: bool = True, **kwargs):
        super(ChebNet, self).__init__(K, dropedge, bias, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = Linear(in_channels, out_channels, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()

    def forward(
        self,
        x,
        edge_index,
        edge_weight,
    ):

        h = super().forward(x, edge_index, edge_weight)
        return self.lin(h)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}-{self.in_channels}, '
                f'{self.out_channels}, K={self.K}')