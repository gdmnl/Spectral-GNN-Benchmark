from typing import Optional, Any

import numpy as np
import torch
from torch import Tensor

from torch_geometric.typing import Adj
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import spmm


class ClenshawConv(MessagePassing):
    r"""Convolutional layer with Chebyshev Polynomials and explicit residual.
    paper: Clenshaw Graph Neural Networks
    ref: https://github.com/yuziGuo/ClenshawGNN/blob/master/models/ChebClenshawNN.py

    Args:
        num_hops (int), hop (int): total and current number of propagation hops.
            hop=0 explicitly handles x without propagation.
        alpha (float): transformation strength.
        cached: whether cache the propagation matrix.
    """
    supports_batch: bool = False
    supports_norm_batch: bool = False
    _cache: Optional[Any]

    def __init__(self,
        num_hops: int = 0,
        hop: int = 0,
        alpha: float = None,
        cached: bool = True,
        **kwargs
    ):
        kwargs.setdefault('aggr', 'add')
        super(ClenshawConv, self).__init__(**kwargs)

        self.num_hops = num_hops
        self.hop = hop
        alpha = alpha or 0
        if alpha == 0:
            self.alpha = 0.5
        else:
            self.alpha = np.log(alpha / (hop + 1) + 1)
            self.alpha = min(self.alpha, 1.0)
        self.beta_init = 1.0 if hop == num_hops else 0.0
        self.beta_init = torch.tensor(self.beta_init)
        self.beta = torch.nn.Parameter(self.beta_init)

        self.cached = cached
        self._cache = None

    def reset_cache(self):
        del self._cache
        self._cache = None

    def reset_parameters(self):
        if hasattr(self.theta, 'reset_parameters'):
            self.theta.reset_parameters()

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
            out_1 (:math:`(|\mathcal{V}|, F)` Tensor): propagation result of k-2
            x_0 (:math:`(|\mathcal{V}|, F)` Tensor): initial input
            prop_mat (Adj): propagation matrix
        """
        return {
            'out': torch.zeros_like(x),
            'out_1': x,
            'x_0': x,
            'prop_mat': self.get_propagate_mat(x, edge_index)}

    def _forward_theta(self, x):
        if callable(self.theta):
            return self.theta(x)
        else:
            return self.theta * x

    def forward(self,
        out: Tensor,
        out_1: Tensor,
        x_0: Tensor,
        prop_mat: Adj,
    ) -> dict:
        r"""
        Args & Returns: (dct): same with output of get_forward_mat()
        """
        # propagate_type: (x: Tensor)
        h = self.propagate(prop_mat, x=out)
        h = self.beta * x_0 + 2. * h - out_1

        h2 = self._forward_theta(h)
        h2 = self.alpha * h + (1 - self.alpha) * h2

        return {
            'out': h2,
            'out_1': out,
            'x_0': x_0,
            'prop_mat': prop_mat}

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(alpha={self.alpha})'
