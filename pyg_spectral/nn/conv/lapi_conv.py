from torch import Tensor

from torch_geometric.typing import Adj
from pyg_spectral.nn.conv.base_mp import BaseMP


class LapiConv(BaseMP):
    r"""Iterative linear filter using the normalized adjacency matrix.
    Used in AdaGNN.

    :paper: AdaGNN: Graph Neural Networks with Adaptive Frequency Response Filter
    :ref: https://github.com/yushundong/AdaGNN/blob/main/layers.py

    Args:
        num_hops, hop, cached: args for :class:`BaseMP`
    """
    # For similar convs supporting batching, use LapSkipConv
    supports_batch: bool = False
    name = lambda _: 'LapiConv'

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

    def forward(self,
        out: Tensor,
        prop: Adj,
    ) -> dict:
        r"""
        Returns:
            out (Tensor): output tensor for
                accumulating propagation results (shape: :math:`(|\mathcal{V}|, F)`)
            prop (Adj): propagation matrix
        """
        # propagate_type: (x: Tensor)
        h = self.propagate(prop, x=out)
        out = out - self._forward_theta(x=h)
        return {'out': out, 'prop': prop}


# TODO: LapSkipConv
