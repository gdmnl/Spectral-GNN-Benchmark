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

    def _get_forward_mat(self, x: Tensor, edge_index: Adj) -> dict:
        return {'out': x,}

    def _get_convolute_mat(self, x: Tensor, edge_index: Adj) -> dict:
        return {'out': x,}

    def _forward(self,
        out: Tensor,
        prop: Adj,
    ) -> dict:
        r"""
        Returns:
            out (:math:`(|\mathcal{V}|, F)` Tensor): output tensor for
                accumulating propagation results
            prop (Adj): propagation matrix
        """
        # propagate_type: (x: Tensor)
        out = self.propagate(prop, x=out)
        self.out_scale = -1.0
        return {'out': out, 'prop': prop}
