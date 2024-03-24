import math
import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops #, get_laplacian
from pyg_spectral.utils import get_laplacian
import torch.nn.functional as F

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

    def __init__(self, K, Init=False, bias=True, **kwargs):
        super(ChebConv2, self).__init__(aggr='add', **kwargs)

        self.K = K
        self.temp = Parameter(torch.Tensor(self.K+1))
        self.Init=Init
        self.reset_parameters()

    def reset_parameters(self):
        self.temp.data.fill_(1.0)

        if self.Init:
            for j in range(self.K+1):
                x_j=math.cos((self.K-j+0.5)*math.pi/(self.K+1))
                self.temp.data[j] = x_j**2

    def forward(self, x, edge_index,edge_weight=None):
        coe_tmp=F.relu(self.temp)
        coe=coe_tmp.clone()

        for i in range(self.K+1):
            coe[i]=coe_tmp[0]*cheby(i,math.cos((self.K+0.5)*math.pi/(self.K+1)))
            for j in range(1,self.K+1):
                x_j=math.cos((self.K-j+0.5)*math.pi/(self.K+1))
                coe[i]=coe[i]+coe_tmp[j]*cheby(i,x_j)
            coe[i]=2*coe[i]/(self.K+1)


        #L=I-D^(-0.5)AD^(-0.5)
        edge_index1, norm1 = get_laplacian(edge_index, edge_weight, normalization=True, dtype=x.dtype, num_nodes=x.size(self.node_dim))

        #L_tilde=L-I
        edge_index_tilde, norm_tilde= add_self_loops(edge_index1, norm1, fill_value=-1.0, num_nodes=x.size(self.node_dim))

        Tx_0=x
        Tx_1=self.propagate(edge_index_tilde,x=x,norm=norm_tilde,size=None)

        out=coe[0]/2*Tx_0+coe[1]*Tx_1

        for i in range(2,self.K+1):
            Tx_2=self.propagate(edge_index_tilde,x=Tx_1,norm=norm_tilde,size=None)
            Tx_2=2*Tx_2-Tx_0
            out=out+coe[i]*Tx_2
            Tx_0,Tx_1 = Tx_1, Tx_2
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)