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

    r"""Convolutional layer with Chebyshev-II Polynomials.
    paper: Convolutional Neural Networks on Graphs with Chebyshev Approximation, Revisited
    ref: https://github.com/ivam-he/ChebNetII/blob/main/main/ChebnetII_pro.py
    
    Args:
        num_hops (int), hop (int): total and current number of propagation hops.
            hop=0 explicitly handles x without propagation.
        alpha (float): decay factor for each hop :math:`1/hop^\alpha`.
        theta (nn.Parameter or nn.Module): transformation of propagation result
            before applying to the output.
        cached: whether cache the propagation matrix.
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
        # FEATURE: `combine_root` as `pyg.nn.conv.SimpleConv`
        kwargs.setdefault('aggr', 'add')
        super(ChebConv2, self).__init__(**kwargs)

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
            for i in range(self.num_hops+1):
                self.temps.data[i] = (math.cos((self.num_hops-i+0.5)*math.pi/(self.num_hops+1)))**2

    def get_propagate_mat(self,
        x: Tensor,
        edge_index: Adj
    ) -> Adj:
        r"""Get matrices for self.propagate(). Called before each forward() with
            same input.

        Args:
            x (Tensor), edge_index (Adj): from pyg.data.Data
        Returns:
            prop_mat (SparseTensor): propagation matrix
        """
        cache = self._cache
        if cache is None:
            # A_norm -> L_norm
            edge_index = get_laplacian(
                edge_index,
                normalization=True,
                dtype=x.dtype)
            # L_norm -> L_norm - I
            diag = edge_index.get_diag()
            edge_index = edge_index.set_diag(diag - 1.0)

            if self.cached:
                self._cache = edge_index
        else:
            edge_index = cache
        return edge_index

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
        if self.hop == 0:
            coe_tmp=F.relu(self.temps)
            coe=coe_tmp.clone()
            
            for i in range(self.num_hops+1):
                coe[i]=coe_tmp[0]*cheby(i,math.cos((self.num_hops+0.5)*math.pi/(self.num_hops+1)))
                for j in range(1,self.num_hops+1):
                    x_j=math.cos((self.num_hops-j+0.5)*math.pi/(self.num_hops+1))
                    coe[i]=coe[i]+coe_tmp[j]*cheby(i,x_j)
                coe[i]=2*coe[i]/(self.num_hops+1)

            return {
                'out': torch.zeros_like(x),
                'x': x,
                'x_1': x,
                'prop_mat': self.get_propagate_mat(x, edge_index),
                'temps': coe}
        else:
            return {
                'out': torch.zeros_like(x),
                'x': x,
                'x_1': x,
                'prop_mat': self.get_propagate_mat(x, edge_index),
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
        prop_mat: Adj,
        temps: Tensor,
    ) -> dict:
        r"""
        Args & Returns: (dct): same with output of get_forward_mat()
        """
        if self.hop == 0:
            out = self._forward_theta(x) * temps[self.hop]/2 # relu propressed.
            return {'out': out, 'x': x, 'x_1': x, 'prop_mat': prop_mat, 'temps': temps}
        elif self.hop == 1:
            h = self.propagate(prop_mat, x=x)
            out += self._forward_theta(h) * temps[self.hop]
            return {'out': out, 'x': h, 'x_1': x, 'prop_mat': prop_mat, 'temps': temps}

        h = self.propagate(prop_mat, x=x)
        h = 2. * h - x_1
        out += self._forward_theta(h) * temps[self.hop]

        return {
            'out': out,
            'x': h,
            'x_1': x,
            'prop_mat': prop_mat,
            'temps': temps}

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'


class ChebConv2Old(MessagePassing):

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
        super(ChebConv2Old, self).__init__(**kwargs)

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

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}(K={})'.format(self.__class__.__name__, self.K)


