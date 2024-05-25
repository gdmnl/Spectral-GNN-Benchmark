from typing import Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from torch_geometric.typing import Adj
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import spmm

from pyg_spectral.utils import get_laplacian


def cheby(i, x):
    if i == 0:
        return 1
    elif i == 1:
        return x
    else:
        T0 = 1
        T1 = x
        for ii in range(2, i+1):
            T2 = 2*x*T1 - T0
            T0, T1 = T1, T2
        return T2


class ChebIIConv(MessagePassing):
    r"""Convolutional layer with Chebyshev-II Polynomials.
    paper: Convolutional Neural Networks on Graphs with Chebyshev Approximation, Revisited
    ref: https://github.com/ivam-he/ChebNetII/blob/main/main/ChebnetII_pro.py

    Args:
        num_hops (int), hop (int): total and current number of propagation hops.
            hop=0 explicitly handles x without propagation.
        alpha (float): decay factor for each hop :math:`1/hop^\alpha`.
        cached: whether cache the propagation matrix.
    """
    supports_batch: bool = False
    supports_norm_batch: bool = False
    _cache: Optional[Any]
    coeffs_data = None

    def __init__(self,
        num_hops: int = 0,
        hop: int = 0,
        cached: bool = True,
        **kwargs
    ):
        kwargs.setdefault('aggr', 'add')
        super(ChebIIConv, self).__init__(**kwargs)

        self.num_hops = num_hops
        self.hop = hop
        if self.hop == 0:
            self.coeffs = nn.Parameter(torch.zeros(self.num_hops+1), requires_grad=True)
            self.__class__.coeffs_data = self.coeffs.data

        self.cached = cached
        self._cache = None

    def reset_cache(self):
        del self._cache
        self._cache = None

    def reset_parameters(self):
        if hasattr(self.theta, 'reset_parameters'):
            self.theta.reset_parameters()
            self.register_buffer('coeff', torch.tensor(1.0))
            self.coeffs_data[self.hop] = self.coeff.data
            if self.hop == 0:
                self.coeffs.requires_grad = False
        else:
            self.coeffs_data[self.hop] = self.theta.data

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
        assert self.num_hops+1 == len(self.coeffs)
        coeffs = F.relu(self.coeffs)
        thetas = coeffs.clone()
        for i in range(self.num_hops+1):
            thetas[i] = coeffs[0] * cheby(i, np.cos((self.num_hops+0.5) * np.pi/(self.num_hops+1)))
            for j in range(1, self.num_hops+1):
                x_j = np.cos((self.num_hops-j+0.5) * np.pi/(self.num_hops+1))
                thetas[i] = coeffs[i] + thetas[j] * cheby(i, x_j)
            thetas[i] = 2*thetas[i]/(self.num_hops+1)
        thetas[0] = thetas[0]/2

        return {
            'out': torch.zeros_like(x),
            'x': x,
            'x_1': x,
            'prop_mat': self.get_propagate_mat(x, edge_index),
            'thetas': thetas}

    def _forward_theta(self, x, thetas):
        if callable(self.theta):
            return self.theta(x) * thetas[self.hop]
        else:
            return thetas[self.hop] * x

    def forward(self,
        out: Tensor,
        x: Tensor,
        x_1: Tensor,
        prop_mat: Adj,
        thetas: Tensor
    ) -> dict:
        r"""
        Args & Returns: (dct): same with output of get_forward_mat()
        """
        if self.hop == 0:
            out = self._forward_theta(x, thetas)
            return {'out': out, 'x': x, 'x_1': x, 'prop_mat': prop_mat, 'thetas': thetas}
        elif self.hop == 1:
            h = self.propagate(prop_mat, x=x)
            out += self._forward_theta(h, thetas)
            return {'out': out, 'x': h, 'x_1': x, 'prop_mat': prop_mat, 'thetas': thetas}

        h = self.propagate(prop_mat, x=x)
        h = 2. * h - x_1
        out += self._forward_theta(h, thetas)

        return {
            'out': out,
            'x': h,
            'x_1': x,
            'prop_mat': prop_mat,
            'thetas': thetas}

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        if len(self.coeffs_data) > 0:
            return f'{self.__class__.__name__}(theta={self.coeffs_data[self.hop]})'
        else:
            if hasattr(self, 'coeff'):
                return f'{self.__class__.__name__}(coeff={self.coeff})'
            else:
                return f'{self.__class__.__name__}(theta={self.theta})'
