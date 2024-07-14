from torch import Tensor

from torch_geometric.typing import Adj
from pyg_spectral.nn.conv.base_mp import BaseMP


class JacobiConv(BaseMP):
    r"""Convolutional layer with Jacobi Polynomials.

    :paper: How Powerful are Spectral Graph Neural Networks
    :ref: https://github.com/GraphPKU/JacobiConv

    Args:
        alpha, beta: hyperparameters in Jacobi polynomials.
        num_hops, hop, cached: args for :class:`BaseMP`
    """
    def __init__(self,
        num_hops: int = 0,
        hop: int = 0,
        alpha: float = None,
        beta: float = None,
        cached: bool = True,
        **kwargs
    ):
        kwargs.setdefault('propagate_mat', 'A')
        super(JacobiConv, self).__init__(num_hops, hop, cached, **kwargs)

        self.alpha = alpha or 1.0
        self.beta = beta or 1.0
        self.l = -1.0
        self.r = 1.0

    def _get_convolute_mat(self, x: Tensor, edge_index: Adj) -> dict:
        return {'x': x, 'x_1': x,}

    def _forward(self,
        x: Tensor,
        x_1: Tensor,
        prop: Adj,
    ) -> dict:
        r"""
        Returns:
            x (Tensor): propagation result of :math:`k-1` (shape: :math:`(|\mathcal{V}|, F)`)
            x_1 (Tensor): propagation result of :math:`k-2` (shape: :math:`(|\mathcal{V}|, F)`)
            prop (Adj): propagation matrix
        """
        a, b, l, r, k = self.alpha, self.beta, self.l, self.r, self.hop
        if self.hop == 0:
            return {'x': x, 'x_1': x, 'prop': prop}
        elif self.hop == 1:
            coeff0 = (a+b+2.) / (r-l)
            coeff1 = (a-b)/2. - coeff0/2.*(l+r)
            # propagate_type: (x: Tensor)
            h = self.propagate(prop, x=x)
            h = coeff0 * h + coeff1 * x
            return {'x': h, 'x_1': x, 'prop': prop}

        cl = 2*k * (k + a + b) * (2*k + a + b - 2)
        c0 = (2*k + a + b - 1) * (2*k + a + b) * (2*k + a + b - 2) / cl
        c1 = (2*k + a + b - 1) * (a**2 - b**2) / cl
        coeff2 = 2 * (k + a - 1) * (k + b -1) * (2*k + a + b) / cl
        # coeff0 = c0 * (2 / (r-l))
        # coeff1 = - c0 * ((r+l) / (r-l)) - c1
        coeff0, coeff1 = c0, c1
        # propagate_type: (x: Tensor)
        h = self.propagate(prop, x=x)
        h = coeff0 * h + coeff1 * x - coeff2 * x_1

        return {'x': h, 'x_1': x, 'prop': prop}

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(alpha={self.alpha}, beta={self.beta})'
