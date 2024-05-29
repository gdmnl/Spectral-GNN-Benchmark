import torch
import torch.nn as nn
from torch import Tensor

from torch_geometric.typing import Adj, OptPairTensor
from torch_geometric.nn.inits import reset
from pyg_spectral.nn.conv.base_mp import BaseMP


class AdjiConv(BaseMP):
    r"""Iterative linear filter using the normalized adjacency matrix for augmented propagation.

    Args:
        alpha (float): decay factor :math:`\alpha(\mathbf{A} + \beta\mathbf{I})`.
            Can be :math:`\alpha < 0`.
        beta (float): scaling for self-loop in adjacency matrix, i.e.
            `improved` in PyG GCNConv and `eps` in GINConv. Can be :math:`\beta < 0`.
            beta = 'var' for learnable beta as parameter.
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
        kwargs.setdefault('propagate_mat', 'A')
        super(AdjiConv, self).__init__(num_hops, hop, cached, **kwargs)

        alpha = alpha or 1.0
        self.register_buffer('alpha', torch.tensor(alpha))
        if beta is None:
            self.register_buffer('beta', torch.tensor(1.0))
        elif isinstance(beta, str) and beta == 'var':
            self.beta = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer('beta', torch.tensor(float(beta)))
        self.beta_init = self.beta.data.item()

    def reset_parameters(self):
        reset(self.theta)
        self.beta.data.fill_(self.beta_init)

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
            'out': x,
            'prop': self.get_propagate_mat(x, edge_index)}

    def forward(self,
        out: Tensor,
        prop: Adj,
    ) -> dict:
        r"""
        Args & Returns: (dct): same with output of get_forward_mat()
        """
        if isinstance(out, Tensor):
            out = (out, out)

        # propagate_type: (x: OptPairTensor)
        h = self.propagate(prop, x=out)

        x_r = out[1]
        if x_r is not None:
            h = h + self.beta * self._forward_theta(x_r)
            h *= self.alpha

        return {
            'out': h,
            'prop': prop}

    def message_and_aggregate(self, adj_t: Adj, x: OptPairTensor) -> Tensor:
        # return spmm(adj_t, x[0], reduce=self.aggr)
        return torch.spmm(adj_t, x[0])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(alpha={self.alpha}, beta={self.beta})'


class Adji2Conv(AdjiConv):
    r"""Iterative linear filter using the 2-hop normalized adjacency matrix for
        augmented propagation.

    Args:
        num_hops (int): total number of propagation hops. NOTE that there are
            only :math:`\text{num_hops} / 2` conv layers.
        hop (int): current number of propagation hops.
        alpha (float): decay factor :math:`\alpha(\mathbf{A} + \beta\mathbf{I})`.
            Can be :math:`\alpha < 0`.
        beta (float): scaling for self-loop in adjacency matrix, i.e.
            `improved` in PyG GCNConv and `eps` in GINConv. Can be :math:`\beta < 0`.
            beta = 'var' for learnable beta as parameter.
        theta (nn.Parameter or nn.Module): transformation of propagation result
            before applying to the output.
        cached: whether cache the propagation matrix.
    """
    def message_and_aggregate(self, adj_t: Adj, x: OptPairTensor) -> Tensor:
        # return spmm(adj_t, spmm(adj_t, x[0], reduce=self.aggr), reduce=self.aggr)
        return torch.spmm(adj_t, torch.spmm(adj_t, x[0]))
