from typing import Optional, Any, Union

import torch
import torch.nn as nn
from torch import Tensor

from torch_geometric.typing import Adj, OptPairTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.utils import spmm


class AdjiConv(MessagePassing):
    r"""Iterative linear filter using the normalized adjacency matrix for augmented propagation.

    Args:
        num_hops (int), hop (int): total and current number of propagation hops.
        alpha (float): decay factor :math:`\alpha(\mathbf{A} + \beta\mathbf{I})`.
            Can be :math:`\alpha < 0`.
        beta (float): scaling for self-loop in adjacency matrix, i.e.
            `improved` in PyG GCNConv and `eps` in GINConv. Can be :math:`\beta < 0`.
            beta = 'var' for learnable beta as parameter.
        theta (nn.Parameter or nn.Module): transformation of propagation result
            before applying to the output.
        cached: whether cache the propagation matrix.
    """
    supports_batch: bool = False
    supports_norm_batch: bool = False
    _cache: Optional[Any]

    def __init__(self,
        num_hops: int = 0,
        hop: int = 0,
        theta: Union[nn.Parameter, nn.Module] = None,
        alpha: float = None,
        beta: float = None,
        cached: bool = True,
        **kwargs
    ):
        kwargs.setdefault('aggr', 'add')
        super(AdjiConv, self).__init__(**kwargs)

        self.num_hops = num_hops
        self.hop = hop
        self.theta = theta
        alpha = alpha or 1.0
        self.register_buffer('alpha', torch.tensor(alpha))
        if beta is None:
            self.register_buffer('beta', torch.tensor(1.0))
        elif isinstance(beta, str) and beta == 'var':
            self.beta = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer('beta', torch.tensor(float(beta)))
        self.beta_init = self.beta.data.item()

        self.cached = cached
        self._cache = None

    def reset_cache(self):
        del self._cache
        self._cache = None

    def reset_parameters(self):
        reset(self.theta)
        self.beta.data.fill_(self.beta_init)

    def get_propagate_mat(self,
        x: Tensor,
        edge_index: Adj
    ) -> Adj:
        r"""Get matrices for self.propagate(). Called before each forward() with
            same input.

        Args:
            x (Tensor), edge_index (Adj): from pyg.data.Data
        Returns:
            prop_mat (SparseTensor): propagation matrix
        """
        cache = self._cache
        if cache is None:
            if self.cached:
                self._cache = edge_index
        else:
            edge_index = cache
        return edge_index

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
            prop_mat (Adj): propagation matrix
        """
        return {
            'out': x,
            'prop_mat': self.get_propagate_mat(x, edge_index)}

    def _forward_theta(self, x):
        if callable(self.theta):
            return self.theta(x)
        else:
            return self.theta * x

    def forward(self,
        out: Tensor,
        prop_mat: Adj,
    ) -> dict:
        r"""
        Args & Returns: (dct): same with output of get_forward_mat()
        """
        if isinstance(out, Tensor):
            out = (out, out)

        # propagate_type: (x: OptPairTensor)
        h = self.propagate(prop_mat, x=out)

        x_r = out[1]
        if x_r is not None:
            h = h + self.beta * self._forward_theta(x_r)
            h *= self.alpha

        return {
            'out': h,
            'prop_mat': prop_mat}

    def message_and_aggregate(self, adj_t: Adj, x: OptPairTensor) -> Tensor:
        return spmm(adj_t, x[0], reduce=self.aggr)

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
        return spmm(adj_t, spmm(adj_t, x[0], reduce=self.aggr), reduce=self.aggr)
