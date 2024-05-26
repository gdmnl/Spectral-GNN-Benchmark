from typing import Optional, Any

import torch
from torch import Tensor

from torch_geometric.typing import Adj
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import spmm


class JacobiConv(MessagePassing):
    r"""Convolutional layer with Jacobi Polynomials.
    paper: How Powerful are Spectral Graph Neural Networks
    ref: https://github.com/GraphPKU/JacobiConv

    Args:
        num_hops (int), hop (int): total and current number of propagation hops.
            hop=0 explicitly handles x without propagation.
        alpha, beta (float): hyperparameters in Jacobi polynomials.
        cached: whether cache the propagation matrix.
    """
    supports_batch: bool = False
    supports_norm_batch: bool = False
    _cache: Optional[Any]

    def __init__(self,
        num_hops: int = 0,
        hop: int = 0,
        alpha: float = None,
        beta: float = None,
        cached: bool = True,
        **kwargs
    ):
        kwargs.setdefault('aggr', 'add')
        super(JacobiConv, self).__init__(**kwargs)

        self.num_hops = num_hops
        self.hop = hop
        self.alpha = alpha or 1.0
        self.beta = beta or 1.0
        self.l = -1.0
        self.r = 1.0

        self.cached = cached
        self._cache = None

    def reset_cache(self):
        del self._cache
        self._cache = None

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
            x (:math:`(|\mathcal{V}|, F)` Tensor): propagation result of k-1
            x_1 (:math:`(|\mathcal{V}|, F)` Tensor): propagation result of k-2
            prop_mat (Adj): propagation matrix
        """
        return {
            'out': torch.zeros_like(x),
            'x': x,
            'x_1': x,
            'prop_mat': self.get_propagate_mat(x, edge_index)}

    def _forward_theta(self, x):
        if callable(self.theta):
            return self.theta(x)
        else:
            return self.theta * x

    def forward(self,
        out: Tensor,
        x: Tensor,
        x_1: Tensor,
        prop_mat: Adj,
    ) -> dict:
        r"""
        Args & Returns: (dct): same with output of get_forward_mat()
        """
        a, b, l, r, k = self.alpha, self.beta, self.l, self.r, self.hop
        if self.hop == 0:
            out = self._forward_theta(x)
            return {'out': out, 'x': x, 'x_1': x, 'prop_mat': prop_mat}
        elif self.hop == 1:
            coeff0 = (a+b+2.) / (r-l)
            coeff1 = (a-b)/2. - coeff0/2.*(l+r)
            # propagate_type: (x: Tensor)
            h = self.propagate(prop_mat, x=x)
            h = coeff0 * h + coeff1 * x
            out += self._forward_theta(h)
            return {'out': out, 'x': h, 'x_1': x, 'prop_mat': prop_mat}

        cl = 2*k * (k + a + b) * (2*k + a + b - 2)
        c0 = (2*k + a + b - 1) * (2*k + a + b) * (2*k + a + b - 2) / cl
        c1 = (2*k + a + b - 1) * (a**2 - b**2) / cl
        coeff2 = 2 * (k + a - 1) * (k + b -1) * (2*k + a + b) / cl
        # coeff0 = c0 * (2 / (r-l))
        # coeff1 = - c0 * ((r+l) / (r-l)) - c1
        coeff0, coeff1 = c0, c1
        # propagate_type: (x: Tensor)
        h = self.propagate(prop_mat, x=x)
        h = coeff0 * h + coeff1 * x - coeff2 * x_1
        out += self._forward_theta(h)

        return {
            'out': out,
            'x': h,
            'x_1': x,
            'prop_mat': prop_mat}

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(alpha={self.alpha}, beta={self.beta})'
