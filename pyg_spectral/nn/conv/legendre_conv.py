from torch import Tensor

from torch_geometric.typing import Adj
from pyg_spectral.nn.conv.base_mp import BaseMP


class LegendreConv(BaseMP):
    r"""Convolutional layer with Legendre Polynomials.

    :paper: How Powerful are Spectral Graph Neural Networks
    :ref: https://github.com/GraphPKU/JacobiConv
    :paper: Improved Modeling and Generalization Capabilities of Graph Neural Networks With Legendre Polynomials
    :ref: https://github.com/12chen20/LegendreNet

    Args:
        num_hops, hop, cached: args for :class:`BaseMP`
    """
    name = lambda _: 'LegendreConv'

    def __init__(self,
        num_hops: int = 0,
        hop: int = 0,
        cached: bool = True,
        **kwargs
    ):
        kwargs.setdefault('propagate_mat', 'A')
        super(LegendreConv, self).__init__(num_hops, hop, cached, **kwargs)

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
        if self.hop == 0:
            return {'x': x, 'x_1': x, 'prop': prop}
        elif self.hop == 1:
            # propagate_type: (x: Tensor)
            h = self.propagate(prop, x=x)
            h = (2. - 1./self.hop) * h
            return {'x': h, 'x_1': x, 'prop': prop}

        # propagate_type: (x: Tensor)
        h = self.propagate(prop, x=x)
        h = (2. - 1./self.hop) * h - (1. - 1./self.hop) * x_1

        return {'x': h, 'x_1': x, 'prop': prop}

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'
