import torch
from torch import Tensor

from torch_geometric.typing import Adj
from pyg_spectral.nn.conv.base_mp import BaseMP


class LegendreConv(BaseMP):
    r"""Convolutional layer with Legendre Polynomials.
    paper: How Powerful are Spectral Graph Neural Networks
    ref: https://github.com/GraphPKU/JacobiConv
    Alternative paper: Improved Modeling and Generalization Capabilities of Graph Neural Networks With Legendre Polynomials
    code: https://github.com/12chen20/LegendreNet

    Args:
        --- BaseMP Args ---
        num_hops (int), hop (int): total and current number of propagation hops.
        cached: whether cache the propagation matrix.
    """
    def __init__(self,
        num_hops: int = 0,
        hop: int = 0,
        cached: bool = True,
        **kwargs
    ):
        kwargs.setdefault('propagate_mat', 'A')
        super(LegendreConv, self).__init__(num_hops, hop, cached, **kwargs)

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
        """
        return {
            'out': torch.zeros_like(x),
            'x': x,
            'x_1': x,
            'prop': self.get_propagate_mat(x, edge_index)}

    def forward(self,
        out: Tensor,
        x: Tensor,
        x_1: Tensor,
        prop: Adj,
    ) -> dict:
        r"""
        Args & Returns: (dct): same with output of get_forward_mat()
        """
        if self.hop == 0:
            out = self._forward_theta(x)
            return {'out': out, 'x': x, 'x_1': x, 'prop': prop}
        elif self.hop == 1:
            # propagate_type: (x: Tensor)
            h = self.propagate(prop, x=x)
            h = (2. - 1./self.hop) * h
            out += self._forward_theta(h)
            return {'out': out, 'x': h, 'x_1': x, 'prop': prop}

        # propagate_type: (x: Tensor)
        h = self.propagate(prop, x=x)
        h = (2. - 1./self.hop) * h - (1. - 1./self.hop) * x_1
        out += self._forward_theta(h)

        return {
            'out': out,
            'x': h,
            'x_1': x,
            'prop': prop}

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'
