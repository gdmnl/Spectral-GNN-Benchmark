import numpy as np
import torch
from torch import Tensor

from torch_geometric.typing import Adj
from torch_geometric.nn.inits import reset
from pyg_spectral.nn.conv.base_mp import BaseMP


class ClenshawConv(BaseMP):
    r"""Convolutional layer with Chebyshev Polynomials and explicit residual.
    paper: Clenshaw Graph Neural Networks
    ref: https://github.com/yuziGuo/ClenshawGNN/blob/master/models/ChebClenshawNN.py

    Args:
        alpha (float): transformation strength.
        --- BaseMP Args ---
        num_hops (int), hop (int): total and current number of propagation hops.
        cached: whether cache the propagation matrix.
    """
    def __init__(self,
        num_hops: int = 0,
        hop: int = 0,
        alpha: float = None,
        cached: bool = True,
        **kwargs
    ):
        kwargs.setdefault('propagate_mat', 'A')
        super(ClenshawConv, self).__init__(num_hops, hop, cached, **kwargs)

        alpha = alpha or 0
        if alpha == 0:
            self.alpha = 0.5
        else:
            self.alpha = np.log(alpha / (hop + 1) + 1)
            self.alpha = min(self.alpha, 1.0)
        self.beta_init = 1.0 if hop == num_hops else 0.0
        self.beta_init = torch.tensor(self.beta_init)
        self.beta = torch.nn.Parameter(self.beta_init)

    def reset_parameters(self):
        reset(self.theta)
        self.beta.data.fill_(self.beta_init)

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
            out_1 (:math:`(|\mathcal{V}|, F)` Tensor): propagation result of k-2
            x_0 (:math:`(|\mathcal{V}|, F)` Tensor): initial input
            prop (Adj): propagation matrix
        """
        return {
            'out': torch.zeros_like(x),
            'out_1': x,
            'x_0': x,
            'prop': self.get_propagate_mat(x, edge_index)}

    def forward(self,
        out: Tensor,
        out_1: Tensor,
        x_0: Tensor,
        prop: Adj,
    ) -> dict:
        r"""
        Args & Returns: (dct): same with output of get_forward_mat()
        """
        # propagate_type: (x: Tensor)
        h = self.propagate(prop, x=out)
        h = self.beta * x_0 + 2. * h - out_1

        h2 = self._forward_theta(h)
        h2 = self.alpha * h + (1 - self.alpha) * h2

        return {
            'out': h2,
            'out_1': out,
            'x_0': x_0,
            'prop': prop}

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(alpha={self.alpha})'
