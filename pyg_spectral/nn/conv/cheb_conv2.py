from typing import Optional, Any, Union

import math
import torch
import torch.nn as nn
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

