from scipy.special import comb
import torch
from torch import Tensor
import torch.nn.functional as F

from torch_geometric.typing import Adj
from pyg_spectral.nn.conv.base_mp import BaseMP


class BernConv(BaseMP):
    r"""Convolutional layer with Bernstein Polynomials.
    We propose a new implementation reducing memory from O(KFn) to O(3Fn).
    paper: BernNet: Learning Arbitrary Graph Spectral Filters via Bernstein Approximation
    ref: https://github.com/ivam-he/BernNet/blob/main/NodeClassification/Bernpro.py

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
        kwargs.setdefault('propagate_mat', 'A+I,L')
        super(BernConv, self).__init__(num_hops, hop, cached, **kwargs)

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
            x (:math:`(|\mathcal{V}|, F)` Tensor): propagation result through (2I-L)
            prop_0 (SparseTensor): L
            prop_1 (SparseTensor): 2I - L
        """
        return {
            'out': torch.zeros_like(x),
            'x': x
            } | self.get_propagate_mat(x, edge_index)

    def _forward_theta(self, x):
        if callable(self.theta):
            return self.theta(x)
        else:
            return self.theta * F.relu(x)

    def forward(self,
        out: Tensor,
        x: Tensor,
        prop_0: Adj,
        prop_1: Adj,
    ) -> dict:
        r"""
        Args & Returns: (dct): same with output of get_forward_mat()
        """
        if self.hop > 0:
            # propagate_type: (x: Tensor)
            x = self.propagate(prop_1, x=x)

        h = x
        # Propagate through L for (K-k) times
        for _ in range(self.num_hops - self.hop):
            # propagate_type: (x: Tensor)
            h = self.propagate(prop_0, x=h)

        h = self._forward_theta(h)
        out += h * comb(self.num_hops, self.num_hops-self.hop) / (2**self.num_hops)

        return {
            'out': out,
            'x': x,
            'prop_0': prop_0,
            'prop_1': prop_1}
