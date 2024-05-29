from torch import Tensor

from torch_geometric.typing import Adj
from pyg_spectral.nn.conv.base_mp import BaseMP


class LapiConv(BaseMP):
    r"""Iterative linear filter using the normalized adjacency matrix.
    Used in AdaGNN.
    paper: AdaGNN: Graph Neural Networks with Adaptive Frequency Response Filter
    ref: https://github.com/yushundong/AdaGNN/blob/main/layers.py

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
        kwargs.setdefault('propagate_mat', 'L')
        super(LapiConv, self).__init__(num_hops, hop, cached, **kwargs)

    def get_forward_mat(self,
        x: Tensor,
        edge_index: Adj,
    ) -> dict:
        r"""Get matrices for self.forward(). Called during forward().

        Args:
            x (Tensor), edge_index (Adj): from pyg.data.Data
        Returns:
            out (:math:`(|\mathcal{V}|, F)` Tensor): current propagation result
            prop (Adj): propagation matrix
        """
        return {
            'out': x,
            'prop': self.get_propagate_mat(x, edge_index)}

    def forward(self,
        out: Tensor,
        prop: Adj,
    ) -> dict:
        r"""
        Args & Returns: (dct): same with output of get_forward_mat()
        """
        # propagate_type: (x: Tensor)
        h = self.propagate(prop, x=out)
        h = out - self._forward_theta(h)

        return {
            'out': h,
            'prop': prop}
