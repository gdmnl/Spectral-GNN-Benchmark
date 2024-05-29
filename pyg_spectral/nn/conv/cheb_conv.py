from torch import Tensor

from torch_geometric.typing import Adj
from pyg_spectral.nn.conv.base_mp import BaseMP


class ChebConv(BaseMP):
    r"""Convolutional layer with Chebyshev Polynomials.
    paper: Convolutional Neural Networks on Graphs with Chebyshev Approximation, Revisited
    ref: https://github.com/ivam-he/ChebNetII/blob/main/main/Chebbase_pro.py

    Args:
        alpha (float): decay factor for each hop :math:`1/hop^\alpha`.
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
        kwargs.setdefault('propagate_mat', 'L-I')
        super(ChebConv, self).__init__(num_hops, hop, cached, **kwargs)
        self.alpha = alpha or 0.0

    def _get_convolute_mat(self, x: Tensor, edge_index: Adj) -> dict:
        return {'x': x, 'x_1': x,}

    def _forward(self,
        x: Tensor,
        x_1: Tensor,
        prop: Adj,
    ) -> dict:
        r"""
        Returns:
            x (:math:`(|\mathcal{V}|, F)` Tensor): propagation result of k-1
            x_1 (:math:`(|\mathcal{V}|, F)` Tensor): propagation result of k-2
            prop (Adj): propagation matrix
        """
        if self.hop == 0:
            # No propagation
            return {'x': x, 'x_1': x, 'prop': prop}
        elif self.hop == 1:
            # propagate_type: (x: Tensor)
            h = self.propagate(prop, x=x)
            return {'x': h, 'x_1': x, 'prop': prop}

        # propagate_type: (x: Tensor)
        h = self.propagate(prop, x=x)
        h = 2. * h - x_1
        self.out_scale = 1.0 / (self.hop ** self.alpha)

        return {'x': h, 'x_1': x, 'prop': prop}

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(alpha={self.alpha})'
