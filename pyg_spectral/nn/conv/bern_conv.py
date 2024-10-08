from scipy.special import comb
from torch import Tensor
import torch.nn.functional as F

from torch_geometric.typing import Adj
from pyg_spectral.nn.conv.base_mp import BaseMP


class BernConv(BaseMP):
    r"""Convolutional layer with Bernstein Polynomials.
    We propose a new implementation reducing memory overhead from
    :math:`O(KFn)` to :math:`O(3Fn)`.

    :paper: BernNet: Learning Arbitrary Graph Spectral Filters via Bernstein Approximation
    :ref: https://github.com/ivam-he/BernNet/blob/main/NodeClassification/Bernpro.py

    Args:
        num_hops, hop, cached: args for :class:`BaseMP`
    """
    name = lambda _: 'BernConv'

    def __init__(self,
        num_hops: int = 0,
        hop: int = 0,
        cached: bool = True,
        **kwargs
    ):
        kwargs.setdefault('propagate_mat', 'A+I,L')
        super(BernConv, self).__init__(num_hops, hop, cached, **kwargs)

    def _forward_theta(self, **kwargs):
        if callable(self.theta):
            return self.theta(kwargs['x'])
        else:
            return self.theta * F.relu(kwargs['x'])

    def _forward(self,
        x: Tensor,
        prop_0: Adj,
        prop_1: Adj,
    ) -> dict:
        r"""
        Returns:
            x (Tensor): propagation result through :math:`2I-L`  (shape: :math:`(|\mathcal{V}|, F)`)
            prop_0 (SparseTensor): :math:`L`
            prop_1 (SparseTensor): :math:`2I-L`
        """
        if self.hop > 0:
            # propagate_type: (x: Tensor)
            x = self.propagate(prop_1, x=x)

        h = x
        # Propagate through L for (K-k) times
        for _ in range(self.num_hops - self.hop):
            # propagate_type: (x: Tensor)
            h = self.propagate(prop_0, x=h)
        self.out_scale = comb(self.num_hops, self.num_hops-self.hop) / (2**self.num_hops)

        return {'x': x, 'prop_0': prop_0, 'prop_1': prop_1}
