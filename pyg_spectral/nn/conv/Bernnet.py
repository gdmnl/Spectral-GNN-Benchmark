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

    def __init__(self, K, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(BernConv, self).__init__(**kwargs)
        assert K > 0
        self.K=K
        self.temp = Parameter(torch.Tensor(self.K+1))
        self.reset_parameters()
    
    def reset_parameters(self):
        self.temp.data.fill_(0.0)
        self.temp.data[0]=1.0

    def forward(self,x,edge_index,edge_weight=None):

        #L=I-D^(-0.5)AD^(-0.5)
        edge_index1, norm1 = get_laplacian(edge_index, edge_weight, normalization=True, dtype=x.dtype, num_nodes=x.size(self.node_dim))

        #2I-L
        edge_index2, norm2 = add_self_loops(edge_index1,-norm1, fill_value=2., num_nodes=x.size(self.node_dim))

        tmp=[]
        tmp.append(x)
        for i in range(self.K):
                x=self.propagate(edge_index2,x=x,norm=norm2,size=None)
                tmp.append(x)

        out=(comb(self.K,0)/(2**self.K))*self.temp[0]*tmp[self.K]

        for i in range(self.K):
                x=tmp[self.K-i-1]
                x=self.propagate(edge_index1,x=x,norm=norm1,size=None)
                for j in range(i):
                        x=self.propagate(edge_index1,x=x,norm=norm1,size=None)

                out=out+(comb(self.K,i+1)/(2**self.K))*self.temp[i+1]*x

        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={}, normalization={})'.format(
            self.__class__.__name__, self.K self.temp)