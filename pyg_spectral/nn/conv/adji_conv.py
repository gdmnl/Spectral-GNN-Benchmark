import torch
import torch.nn as nn
from torch import Tensor

from torch_geometric.typing import Adj
from torch_geometric.nn.inits import reset
from pyg_spectral.nn.conv.base_mp import BaseMP


class AdjiConv(BaseMP):
    r"""Iterative linear filter using the normalized adjacency matrix for augmented propagation.

    Args:
        alpha: decay factor :math:`\alpha(\mathbf{A} + \beta\mathbf{I})`.
            Can be :math:`\alpha < 0`.
        beta: scaling for skip connection, i.e., self-loop in adjacency
            matrix, i.e. :obj:`improved` in :class:`torch_geometric.nn.conv.GCNConv`
            and :obj:`eps` in :class:`torch_geometric.nn.conv.GINConv`.
            Can be :math:`\beta < 0`.
            ``beta = 'var'`` for learnable beta as parameter.
        num_hops, hop, cached: args for :class:`BaseMP`
    """
    # For similar convs supporting batching, use AdjConv or AdjSkipConv
    supports_batch: bool = False
    name = lambda _: 'AdjiConv'
    pargs = ['alpha', 'beta']
    param = {
        'alpha': ('float', (0.00, 1.00), {'step': 0.01}, lambda x: round(x, 2)),
        'beta':  ('float', (0.01, 2.00), {'step': 0.01}, lambda x: round(x, 2)),
    }

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
            self.beta = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer('beta', torch.tensor(float(beta)))
        self.beta_init = self.beta.data.item()

    def reset_parameters(self):
        reset(self.theta)
        self.beta.data.fill_(self.beta_init)

    def _get_forward_mat(self, x: Tensor, edge_index: Adj) -> dict:
        return {'out': x,}

    def _get_convolute_mat(self, x: Tensor, edge_index: Adj) -> dict:
        return {'out': x,}

    def forward(self,
        out: Tensor,
        prop: Adj,
    ) -> dict:
        r"""Overwrite forward method.

        Returns:
            out (Tensor): output tensor for accumulating propagation results
                (shape: :math:`(|\mathcal{V}|, F)`)
            prop (Adj): propagation matrix
        """
        # propagate_type: (x: Tensor)
        h = self.propagate(prop, x=out)
        out = h + self.beta * out
        out = self._forward_theta(x=out) * self.alpha
        return {'out': out, 'prop': prop}

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(alpha={self.alpha}, beta={self.beta})'


class Adji2Conv(AdjiConv):
    r"""Iterative linear filter using the 2-hop normalized adjacency matrix for
    augmented propagation.

    Args:
        num_hops: total number of propagation hops. NOTE that there are
            only :math:`\text{num_hops} / 2` conv layers.
        alpha: decay factor :math:`\alpha(\mathbf{A} + \beta\mathbf{I})`.
            Can be :math:`\alpha < 0`.
        beta: scaling for skip connection, i.e., self-loop in adjacency
            matrix, i.e. :obj:`improved` in :class:`torch_geometric.nn.conv.GCNConv`
            and :obj:`eps` in :class:`torch_geometric.nn.conv.GINConv`.
            Can be :math:`\beta < 0`.
            ``beta = 'var'`` for learnable beta as parameter.
        hop, cached: args for :class:`BaseMP`
    """
    name = lambda _: 'Adji2Conv'

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        r""" Perform 2-hop propagation.
        """
        # return spmm(adj_t, spmm(adj_t, x, reduce=self.aggr), reduce=self.aggr)
        return torch.spmm(adj_t, torch.spmm(adj_t, x))


class AdjSkipConv(BaseMP):
    r"""Iterative linear filter with skip connection.

    Args:
        alpha: decay factor :math:`\alpha(\mathbf{A} + \beta\mathbf{I})`.
            Can be :math:`\alpha < 0`.
        beta: scaling for skip connection, i.e., self-loop in adjacency
            matrix, i.e. :obj:`improved` in :class:`torch_geometric.nn.conv.GCNConv`
            and :obj:`eps` in :class:`torch_geometric.nn.conv.GINConv`.
            Can be :math:`\beta < 0`.
            ``beta = 'var'`` for learnable beta as parameter.
        num_hops, hop, cached: args for :class:`BaseMP`
    """
    name = lambda _: 'AdjSkipConv'
    pargs = ['alpha', 'beta']
    param = {
        'alpha': ('float', (0.00, 1.00), {'step': 0.01}, lambda x: round(x, 2)),
        'beta':  ('float', (0.01, 2.00), {'step': 0.01}, lambda x: round(x, 2)),
    }

    def __init__(self,
        num_hops: int = 0,
        hop: int = 0,
        alpha: float = None,
        beta: float = None,
        cached: bool = True,
        **kwargs
    ):
        kwargs.setdefault('propagate_mat', 'A')
        super(AdjSkipConv, self).__init__(num_hops, hop, cached, **kwargs)

        alpha = alpha or 1.0
        self.register_buffer('alpha', torch.tensor(alpha))
        if beta is None:
            self.beta = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer('beta', torch.tensor(float(beta)))
        self.beta_init = self.beta.data.item()

    def reset_parameters(self):
        reset(self.theta)
        self.beta.data.fill_(self.beta_init)

    def _get_forward_mat(self, x: Tensor, edge_index: Adj) -> dict:
        return {'out': x, 'h': x}

    def _get_convolute_mat(self, x: Tensor, edge_index: Adj) -> dict:
        return {'out': x, 'h': x}

    def _forward_out(self, **kwargs) -> Tensor:
        r"""
        Returns:
            out (Tensor): output tensor for accumulating propagation results
                (shape: :math:`(|\mathcal{V}|, F)`)
        """
        out, h = kwargs['out'], kwargs['h']
        out = h + self.beta * out
        out = self._forward_theta(x=out) * self.alpha
        return out

    def _forward(self,
        out: Tensor,
        h: Tensor,
        prop: Adj,
    ) -> dict:
        r"""
        Returns:
            out (Tensor): output tensor for accumulating propagation results
                (shape: :math:`(|\mathcal{V}|, F)`)
            prop (Adj): propagation matrix
        """
        # propagate_type: (x: Tensor)
        h = self.propagate(prop, x=out)
        return {'out': out, 'h': h, 'prop': prop}

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(alpha={self.alpha}, beta={self.beta})'


class AdjSkip2Conv(AdjSkipConv):
    r"""Iterative linear filter with 2-hop propagation and skip connection.

    Args:
        alpha: decay factor :math:`\alpha(\mathbf{A} + \beta\mathbf{I})`.
            Can be :math:`\alpha < 0`.
        beta: scaling for skip connection, i.e., self-loop in adjacency
            matrix, i.e. :obj:`improved` in :class:`torch_geometric.nn.conv.GCNConv`
            and :obj:`eps` in :class:`torch_geometric.nn.conv.GINConv`.
            Can be :math:`\beta < 0`.
            ``beta = 'var'`` for learnable beta as parameter.
        num_hops, hop, cached: args for :class:`BaseMP`
    """
    name = lambda _: 'AdjSkip2Conv'

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        r""" Perform 2-hop propagation.
        """
        # return spmm(adj_t, spmm(adj_t, x, reduce=self.aggr), reduce=self.aggr)
        return torch.spmm(adj_t, torch.spmm(adj_t, x))


class AdjResConv(BaseMP):
    r"""Iterative linear filter with residual connection.

    Args:
        alpha: decay factor :math:`\alpha(\mathbf{A} + \beta\mathbf{I})`.
            Can be :math:`\alpha < 0`.
        beta: scaling for skip connection, i.e., self-loop in adjacency
            matrix, i.e. :obj:`improved` in :class:`torch_geometric.nn.conv.GCNConv`
            and :obj:`eps` in :class:`torch_geometric.nn.conv.GINConv`.
            Can be :math:`\beta < 0`.
            ``beta = 'var'`` for learnable beta as parameter.
        num_hops, hop, cached: args for :class:`BaseMP`
    """
    name = lambda _: 'AdjResConv'
    pargs = ['alpha', 'beta']
    param = {
        'alpha': ('float', (0.00, 1.00), {'step': 0.01}, lambda x: round(x, 2)),
        'beta':  ('float', (0.01, 2.00), {'step': 0.01}, lambda x: round(x, 2)),
    }

    def __init__(self,
        num_hops: int = 0,
        hop: int = 0,
        alpha: float = None,
        beta: float = None,
        cached: bool = True,
        **kwargs
    ):
        kwargs.setdefault('propagate_mat', 'A')
        super(AdjResConv, self).__init__(num_hops, hop, cached, **kwargs)

        alpha = alpha or 1.0
        self.register_buffer('alpha', torch.tensor(alpha))
        if beta is None:
            self.beta = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer('beta', torch.tensor(float(beta)))
        self.beta_init = self.beta.data.item()

    def reset_parameters(self):
        reset(self.theta)
        self.beta.data.fill_(self.beta_init)

    def _get_forward_mat(self, x: Tensor, edge_index: Adj) -> dict:
        return {'out': torch.zeros_like(x), 'x_0': x}

    def _get_convolute_mat(self, x: Tensor, edge_index: Adj) -> dict:
        return {'out': torch.zeros_like(x), 'x_0': x}

    def _forward_out(self, **kwargs) -> Tensor:
        r"""
        Returns:
            out (Tensor): output tensor for accumulating propagation results
                (shape: :math:`(|\mathcal{V}|, F)`)
        """
        out, x_0 = kwargs['out'], kwargs['x_0']
        out = x_0 + self.beta * self._forward_theta(x=out)
        out *= self.alpha
        return out

    def _forward(self,
        out: Tensor,
        x_0: Tensor,
        prop: Adj,
    ) -> dict:
        r"""
        Returns:
            out (Tensor): output tensor for accumulating propagation results
                (shape: :math:`(|\mathcal{V}|, F)`)
            x_0 (Tensor): initial input (shape: :math:`(|\mathcal{V}|, F)`)
            prop (Adj): propagation matrix
        """
        if self.hop > 0:
            # propagate_type: (x: Tensor)
            out = self.propagate(prop, x=out)
        elif self.comp_scheme is not None:
            out = x_0.clone()
        return {'out': out, 'x_0': x_0, 'prop': prop}

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(alpha={self.alpha}, beta={self.beta})'
