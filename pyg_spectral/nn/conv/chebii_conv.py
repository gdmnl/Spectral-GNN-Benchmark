import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from torch_geometric.typing import Adj
from pyg_spectral.nn.conv.base_mp import BaseMP


def cheby(i, x):
    if i == 0:
        return 1
    elif i == 1:
        return x
    else:
        T0 = 1
        T1 = x
        for _ in range(2, i+1):
            T2 = 2*x*T1 - T0
            T0, T1 = T1, T2
        return T2


class ChebIIConv(BaseMP):
    r"""Convolutional layer with Chebyshev-II Polynomials.
    paper: Convolutional Neural Networks on Graphs with Chebyshev Approximation, Revisited
    ref: https://github.com/ivam-he/ChebNetII/blob/main/main/ChebnetII_pro.py

    Args:
        alpha (float): decay factor for each hop :math:`1/hop^\alpha`.
        --- BaseMP Args ---
        num_hops (int), hop (int): total and current number of propagation hops.
        cached: whether cache the propagation matrix.
    """
    coeffs_data = None

    def __init__(self,
        num_hops: int = 0,
        hop: int = 0,
        cached: bool = True,
        **kwargs
    ):
        kwargs.setdefault('propagate_mat', 'L-I')
        super(ChebIIConv, self).__init__(num_hops, hop, cached, **kwargs)

        if self.hop == 0:
            self.coeffs = nn.Parameter(torch.zeros(self.num_hops+1), requires_grad=True)
            self.__class__.coeffs_data = self.coeffs.data

    def reset_parameters(self):
        if hasattr(self.theta, 'reset_parameters'):
            self.theta.reset_parameters()
            self.register_buffer('coeff', torch.tensor(1.0))
            self.coeffs_data[self.hop] = self.coeff.data
            if self.hop == 0:
                self.coeffs.requires_grad = False
        else:
            self.coeffs_data[self.hop] = self.theta.data

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
            prop (Adj): propagation matrix
            thetas (Tensor): learnable/fixed (wrt decoupled/iterative model)
                scalar parameters representing cheb(x)
        """
        # assert self.num_hops+1 == len(self.coeffs)
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
            'prop': self.get_propagate_mat(x, edge_index),
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
        prop: Adj,
        thetas: Tensor
    ) -> dict:
        r"""
        Args & Returns: (dct): same with output of get_forward_mat()
        """
        if self.hop == 0:
            out = self._forward_theta(x, thetas)
            return {'out': out, 'x': x, 'x_1': x, 'prop': prop, 'thetas': thetas}
        elif self.hop == 1:
            h = self.propagate(prop, x=x)
            out += self._forward_theta(h, thetas)
            return {'out': out, 'x': h, 'x_1': x, 'prop': prop, 'thetas': thetas}

        h = self.propagate(prop, x=x)
        h = 2. * h - x_1
        out += self._forward_theta(h, thetas)

        return {
            'out': out,
            'x': h,
            'x_1': x,
            'prop': prop,
            'thetas': thetas}

    def __repr__(self) -> str:
        if len(self.coeffs_data) > 0:
            return f'{self.__class__.__name__}(theta={self.coeffs_data[self.hop]})'
        else:
            if hasattr(self, 'coeff'):
                return f'{self.__class__.__name__}(coeff={self.coeff})'
            else:
                return f'{self.__class__.__name__}(theta={self.theta})'
