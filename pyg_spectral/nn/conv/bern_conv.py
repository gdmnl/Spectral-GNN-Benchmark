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
from torch_geometric.utils import spmm, add_self_loops, remove_self_loops #,get_laplacian
from pyg_spectral.utils import get_laplacian
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch.nn.functional as F
from scipy.special import comb

class BernConv(MessagePassing):

    r"""Convolutional layer with Bernstein Polynomials.
    paper: BernNet: Learning Arbitrary Graph Spectral Filters via Bernstein Approximation
    ref: https://github.com/ivam-he/BernNet/blob/main/NodeClassification/Bernpro.py
    
    Args:
        K (int, optional): Number of iterations :math:`K`.
    """
    supports_batch: bool = False
    supports_norm_batch: bool = False
    _cache: Optional[Any]

    def __init__(self, 
        num_hops: int = 0,
        hop: int = 0,
        theta: Union[nn.Parameter, nn.Module] = None,
        alpha: float = -1.0,
        cached: bool = True,
        **kwargs
    ):
        kwargs.setdefault('aggr', 'add')
        super(BernConv, self).__init__(**kwargs)

        self.num_hops = num_hops
        self.hop = hop
        self.theta = theta
        self.alpha = 0.0 if alpha < 0 else alpha  #NOTE: set actual alpha default here
        self.temps = None

        self.cached = cached
        self._cache = None

    def reset_cache(self):
        del self._cache
        self._cache = None
    
    def reset_parameters(self):
        if hasattr(self.theta, 'reset_parameters'):
            self.theta.reset_parameters()
        if self.hop == 0:
            self.temps = nn.Parameter(torch.ones(self.num_hops+1))
            self.temps.data.fill_(1.0)

    def get_propagate_mat(self,
        x: Tensor,
        edge_index: Adj,
    ) -> [Adj Adj]:
        r"""Get matrices for self.propagate(). Called before each forward() with
            same input.

        Args:
            x (Tensor), edge_index (Adj): from pyg.data.Data
        Returns:
            List [prop_mat, prop_mat],
            prop_mat (SparseTensor): propagation matrix
        """
        cache = self._cache
        # A_norm -> L_norm
        edge_index = get_laplacian(
            edge_index,
            normalization=True,
            dtype=x.dtype)
        # L_norm -> 2I - L_norm
        diag = edge_index.get_diag()
        prop_mats = [edge_index, edge_index.set_diag(2.0 - diag)]

        return prop_mats #[edge_index_1, edge_index_2]

    def get_forward_mat(self,
        x: Tensor,
        edge_index: Adj,
    ) -> dict:
        r"""Get matrices for self.forward(). Called during forward().

        Args:
            x (Tensor), edge_index (Adj): from pyg.data.Data
        Returns:
            out (:math:`(|\mathcal{V}|, F)` Tensor): output tensor for
                accumulating propagation results
            x (:math:`(|\mathcal{V}|, F)` Tensor): propagation result of k-1
            x_1 (:math:`(|\mathcal{V}|, F)` Tensor): propagation result of k-2
            prop_mat (Adj): propagation matrix
        """
        prop_mats = self.get_propagate_mat(x, edge_index)
        return {
            'out': torch.zeros_like(x),
            'x': x,
            'x_1': x,
            'prop_mat_1': prop_mats[0],
            'prop_mat_2': prop_mats[1],
            'temps': self.temps}

    def _forward_theta(self, x):
        if callable(self.theta):
            return self.theta(x)
        else:
            return self.theta * x

    def forward(self,
        out: Tensor,
        x: Tensor,
        x_1: Tensor,
        prop_mat_1: Adj,
        prop_mat_2: Adj,
        temps: Tensor,
    ) -> dict:
        r"""
        Args & Returns: (dct): same with output of get_forward_mat()
        """
        if self.hop == 0:
            out = self._forward_theta(x) * F.relu(temps[self.hop])
            return {'out': out, 'x': x, 'x_1': x, 'prop_mat_1': prop_mat_1, 'prop_mat_2': prop_mat_2, 'temps': temps}
        #elif self.hop == 1:
        #    h = self._forward_theta(self.propagate(prop_mat_2, x=x))
        #    out += comb(1,1)/(2**1) * self._forward_theta(self.propagate(prop_mat_1, self._forward_theta(h)))
        #    return {'out': out, 'x': h, 'x_1': x, 'prop_mat_1': prop_mat_1, 'prop_mat_2': prop_mat_2}

        x = self._forward_theta(self.propagate(prop_mat_2, x=x)) # corresponding to tmps.
        
        if self.hop == self.hops:
            out += (comp(self.num_hops, self.num_hops-self.hop+1) / (2**self.num_hops)) * x * F.relu(temps[self.hop])
            return {
                'out': out,
                'x': h,
                'x_1': x,
                'prop_mat_1': prop_mat_1,
                'prop_mat_2': prop_mat_2,
                'temps': temps}

        x_1 = self._forward_theta(self.propagate(prop_mat_1, x=x))

        for j in range(self.num_hops-self.hop):
            x_1 = self._forward_theta(self.propagate(prop_mat_1, x=x_1))
            # ? x_1 = self.propagate(prop_mat_1, x=h)

        out += comp(self.num_hops, self.num_hops-self.hop+1) / (2**self.num_hops) * x_1 * F.relu(temps[self.hop])

        return {
            'out': out,
            'x': x,
            'x_1': x,
            'prop_mat_1': prop_mat_1,
            'prop_mat_2': prop_mat_2,
            'temps': temps}

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'


class BernConvOld(MessagePassing):

    r"""Convolutional layer with Bernstein Polynomials.
    paper: BernNet: Learning Arbitrary Graph Spectral Filters via Bernstein Approximation
    ref: https://github.com/ivam-he/BernNet/blob/main/NodeClassification/Bernpro.py
    
    Args:
        K (int, optional): Number of iterations :math:`K`.
    """
    supports_batch: bool = False
    supports_norm_batch: bool = False
    _cache: Optional[Any]

    def __init__(self, K: int = 0, cached: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(BernConvOld, self).__init__(**kwargs)

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