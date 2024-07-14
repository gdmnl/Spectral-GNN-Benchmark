from torch import Tensor

from torch_geometric.typing import Adj
from pyg_spectral.nn.conv.base_mp import BaseMP


class AdjConv(BaseMP):
    r"""Linear filter using the normalized adjacency matrix for propagation.

    Args:
        alpha: additional scaling for self-loop in adjacency matrix
            :math:`\mathbf{A} + \alpha\mathbf{I}`, i.e. :obj:`improved` in
            :class:`torch_geometric.nn.conv.GCNConv`.
        num_hops, hop, cached: args for :class:`BaseMP`
    """
    def __init__(self,
        num_hops: int = 0,
        hop: int = 0,
        alpha: float = None,
        cached: bool = True,
        **kwargs
    ):
        kwargs.setdefault('propagate_mat', 'A+alpha*I')
        super(AdjConv, self).__init__(num_hops, hop, cached, **kwargs)
        self.alpha = alpha or 0.0

    def _forward(self,
        x: Tensor,
        prop: Adj,
    ) -> tuple:
        r"""
        Returns:
            x (Tensor): current propagation result (shape: :math:`(|\mathcal{V}|, F)`)
            prop (Adj): propagation matrix
        """
        if self.hop == 0 and not callable(self.theta):
            # No propagation
            return {'x': x, 'prop': prop}

        # NOTE: conduct propagation first, then transformation (different to GCN)
        # propagate_type: (x: Tensor)
        x = self.propagate(prop, x=x)

        return {'x': x, 'prop': prop}


class AdjDiffConv(AdjConv):
    r"""Linear filter using the normalized adjacency matrix for propagation.
    Preprocess the feature by distinguish matrix :math:`\beta\mathbf{L} + \mathbf{I}`.

    Args:
        alpha: additional scaling for self-loop in adjacency matrix
            :math:`\mathbf{A} + \alpha\mathbf{I}`, i.e. :obj:`improved` in
            :class:`torch_geometric.nn.conv.GCNConv`.
        beta: scaling for self-loop in distinguish matrix
            :math:`\beta\mathbf{L} + \mathbf{I}`
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
        super(AdjDiffConv, self).__init__(num_hops, hop, alpha, cached, **kwargs)
        self.beta = beta or 1.0

    def _forward(self,
        x: Tensor,
        prop: Adj,
    ) -> dict:
        r"""
        Returns:
            x (Tensor): current propagation result (shape: :math:`(|\mathcal{V}|, F)`)
            prop (Adj): propagation matrix
        """
        if self.hop == 0:
            # I + beta * L -> (1+beta) * I - beta * A
            h = self.propagate(prop, x=x)
            h = (1 + self.beta) * x - self.beta * h
            return {'x': h, 'prop': prop}

        # propagate_type: (x: Tensor)
        x = self.propagate(prop, x=x)
        return {'x': x, 'prop': prop}
