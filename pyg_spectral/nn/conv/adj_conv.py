from torch import Tensor

from torch_geometric.typing import Adj
from pyg_spectral.nn.conv.base_mp import BaseMP


class AdjConv(BaseMP):
    r"""Linear filter using the normalized adjacency matrix for propagation.

    Args:
        beta: additional scaling for self-loop in adjacency matrix
            :math:`\mathbf{A} + \beta\mathbf{I}`, i.e. :obj:`improved` in
            :class:`torch_geometric.nn.conv.GCNConv`.
        num_hops, hop, cached: args for :class:`BaseMP`
    """
    name = lambda _: 'AdjConv'
    pargs = ['beta']
    param = {'beta': ('float', (0.01, 2.00), {'step': 0.01}, lambda x: round(x, 2))}

    def __init__(self,
        num_hops: int = 0,
        hop: int = 0,
        beta: float = None,
        cached: bool = True,
        **kwargs
    ):
        kwargs.setdefault('propagate_mat', 'A+beta*I')
        super(AdjConv, self).__init__(num_hops, hop, cached, **kwargs)
        self.beta = beta or 0.0

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
    Preprocess the feature by distinguish matrix :math:`\alpha\mathbf{L} + \mathbf{I}`.

    Args:
        alpha: scaling for self-loop in distinguish matrix
            :math:`\alpha\mathbf{L} + \mathbf{I}`
        beta: additional scaling for self-loop in adjacency matrix
            :math:`\mathbf{A} + \beta\mathbf{I}`, i.e. :obj:`improved` in
            :class:`torch_geometric.nn.conv.GCNConv`.
        num_hops, hop, cached: args for :class:`BaseMP`
    """
    name = lambda _: 'AdjDiffConv'
    pargs = ['alpha']
    param = {'alpha': ('float', (0.0, 1.0), {'step': 0.01}, lambda x: round(x, 2))}

    def __init__(self,
        num_hops: int = 0,
        hop: int = 0,
        alpha: float = None,
        beta: float = None,
        cached: bool = True,
        **kwargs
    ):
        super(AdjDiffConv, self).__init__(num_hops, hop, beta, cached, **kwargs)
        self.alpha = alpha or 1.0

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
            # I + alpha * L -> (1+alpha) * I - alpha * A
            h = self.propagate(prop, x=x)
            h = (1 + self.alpha) * x - self.alpha * h
            return {'x': h, 'prop': prop}

        # propagate_type: (x: Tensor)
        x = self.propagate(prop, x=x)
        return {'x': x, 'prop': prop}
