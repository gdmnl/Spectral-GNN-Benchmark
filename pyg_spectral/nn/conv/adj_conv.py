import torch
from torch import Tensor

from torch_geometric.typing import Adj
from pyg_spectral.nn.conv.base_mp import BaseMP


class AdjConv(BaseMP):
    r"""Linear filter using the normalized adjacency matrix for propagation.

    Args:
        alpha (float): additional scaling for self-loop in adjacency matrix
            :math:`\mathbf{A} + \alpha\mathbf{I}`, i.e. `improved` in PyG GCNConv.
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
        kwargs.setdefault('propagate_mat', 'A+alpha*I')
        super(AdjConv, self).__init__(num_hops, hop, cached, **kwargs)
        self.alpha = alpha or 0.0

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
            x (:math:`(|\mathcal{V}|, F)` Tensor): current propagation result
            prop (Adj): propagation matrix
        """
        return {
            'out': torch.zeros_like(x),
            'x': x,
            'prop': self.get_propagate_mat(x, edge_index)}

    def forward(self,
        out: Tensor,
        x: Tensor,
        prop: Adj,
    ) -> dict:
        r"""
        Args & Returns: (dct): same with output of get_forward_mat()
        """
        if self.hop == 0 and not callable(self.theta):
            # No propagation
            out = self._forward_theta(x)
            return {'out': out, 'x': x, 'prop': prop}

        # propagate_type: (x: Tensor)
        x = self.propagate(prop, x=x)

        # NOTE: different to GCN, here conduct propagation first, then transformation
        out += self._forward_theta(x)

        return {
            'out': out,
            'x': x,
            'prop': prop}


class AdjDiffConv(AdjConv):
    r"""Linear filter using the normalized adjacency matrix for propagation.
        Preprocess the feature by distinguish matrix :math:`\beta\mathbf{L} + \mathbf{I}`.

    Args:
        alpha (float): additional scaling for self-loop in adjacency matrix
            :math:`\mathbf{A} + \alpha\mathbf{I}`, i.e. `improved` in PyG GCNConv.
        beta (float): scaling for self-loop in distinguish matrix
            :math:`\beta\mathbf{L} + \mathbf{I}`
        --- BaseMP Args ---
        num_hops (int), hop (int): total and current number of propagation hops.
        cached: whether cache the propagation matrix.
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

    def forward(self,
        out: Tensor,
        x: Tensor,
        prop: Adj,
    ) -> dict:
        r"""
        Args & Returns: (dct): same with output of get_forward_mat()
        """
        if self.hop == 0:
            # I + beta * L -> (1+beta) * I - beta * A
            out = self.propagate(prop, x=x)
            out = self._forward_theta(x)
            out = (1 + self.beta) * x - self.beta * out
            return {'out': out, 'x': out.clone(), 'prop': prop}

        # propagate_type: (x: Tensor)
        x = self.propagate(prop, x=x)

        out += self._forward_theta(x)

        return {
            'out': out,
            'x': x,
            'prop': prop}
