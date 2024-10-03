import numpy as np
import torch
from torch import Tensor

from torch_geometric.typing import Adj
from torch_geometric.nn.inits import reset
from pyg_spectral.nn.conv.base_mp import BaseMP


class ClenshawConv(BaseMP):
    r"""Convolutional layer with Chebyshev Polynomials and explicit residual.

    :paper: Clenshaw Graph Neural Networks
    :ref: https://github.com/yuziGuo/ClenshawGNN/blob/master/models/ChebClenshawNN.py

    Args:
        alpha: transformation strength.
        num_hops, hop, cached: args for :class:`BaseMP`
    """
    name = lambda _: 'ClenshawConv'
    pargs = ['alpha']
    param = {'alpha': ('float', (1e-2, 10), {'log': True}, lambda x: float(f'{x:.2e}')),}

    def __init__(self,
        num_hops: int = 0,
        hop: int = 0,
        alpha: float = None,
        cached: bool = True,
        **kwargs
    ):
        kwargs.setdefault('propagate_mat', 'A')
        super(ClenshawConv, self).__init__(num_hops, hop, cached, **kwargs)

        if alpha is None:
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

    def _get_forward_mat(self, x: Tensor, edge_index: Adj) -> dict:
        return {'out': torch.zeros_like(x), 'out_1': x, 'x_0': x}

    def _get_convolute_mat(self, x: Tensor, edge_index: Adj) -> dict:
        return {'out': torch.zeros_like(x), 'out_1': x, 'x_0': x}

    def _forward_out(self, **kwargs) -> Tensor:
        r"""
        Returns:
            out (Tensor): output tensor for accumulating propagation results
                (shape: :math:`(|\mathcal{V}|, F)`)
        """
        out, x_0 = kwargs['out'], kwargs['x_0']
        out = out + self.beta * x_0

        res = self._forward_theta(x=out)
        res = self.alpha * out + (1 - self.alpha) * res
        return res

    def _forward(self,
        out: Tensor,
        out_1: Tensor,
        x_0: Tensor,
        prop: Adj,
    ) -> dict:
        r"""
        Returns:
            out (Tensor): output tensor for accumulating propagation results
                (shape: :math:`(|\mathcal{V}|, F)`)
            out_1 (Tensor): propagation result of :math:`k-2` (shape: :math:`(|\mathcal{V}|, F)`)
            x_0 (Tensor): initial input (shape: :math:`(|\mathcal{V}|, F)`)
            prop (Adj): propagation matrix
        """
        # propagate_type: (x: Tensor)
        h = self.propagate(prop, x=out)
        h = 2. * h - out_1
        return {'out': h, 'out_1': out, 'x_0': x_0, 'prop': prop}

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(alpha={self.alpha})'
