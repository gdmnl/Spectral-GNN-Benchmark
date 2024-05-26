from typing import Optional, Any, Union

import copy
import torch
import torch.nn as nn
from torch import Tensor

from torch_geometric.typing import Adj
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import spmm


class FavardConv(MessagePassing):
    r"""Convolutional layer with basis in Favard's Theorem.
    paper: Graph Neural Networks with Learnable and Optimal Polynomial Bases
    ref: https://github.com/yuziGuo/FarOptBasis/blob/1e3fdac8ea03b8c98110f740cd79afea9fd4831b/layers/FavardNormalConv.py

    Args:
        num_hops (int), hop (int): total and current number of propagation hops.
            hop=0 explicitly handles x without propagation.
        cached: whether cache the propagation matrix.
    """
    supports_batch: bool = False
    supports_norm_batch: bool = False
    _cache: Optional[Any]

    def __init__(self,
        num_hops: int = 0,
        hop: int = 0,
        cached: bool = True,
        **kwargs
    ):
        kwargs.setdefault('aggr', 'add')
        super(FavardConv, self).__init__(**kwargs)

        self.num_hops = num_hops
        self.hop = hop

        self.cached = cached
        self._cache = None

    def reset_cache(self):
        del self._cache
        self._cache = None

    def reset_parameters(self):
        if hasattr(self.theta, 'reset_parameters'):
            self.theta.reset_parameters()
            self.alpha_pos = copy.deepcopy(self.theta)
            self.beta  = copy.deepcopy(self.theta)
        else:
            self.register_parameter('alpha', nn.Parameter(self.theta))
            self.register_parameter('beta', nn.Parameter(self.theta))

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
            alpha_1: parameter for k-1
        """
        return {
            'out': torch.zeros_like(x),
            'x': x,
            'x_1': torch.zeros_like(x),
            'prop_mat': self.get_propagate_mat(x, edge_index),
            'alpha_1': None}

    @property
    def alpha_pos(self):
        return torch.clamp(self.alpha, min=1e-2)

    def _multiply_coeff(self, x, coeff):
        if callable(coeff):
            return coeff(x)
        else:
            return coeff * x

    def _devide_coeff(self, x, coeff):
        if callable(coeff):
            assert hasattr(coeff, 'weight')
            # use L2 norm of weight
            return x / torch.norm(coeff.weight)
        else:
            return x / coeff

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
        alpha_1: Union[nn.Parameter, nn.Module]
    ) -> dict:
        r"""
        Args & Returns: (dct): same with output of get_forward_mat()
        """
        if self.hop == 0:
            h = self._devide_coeff(x, self.alpha_pos)
            out = self._forward_theta(h)
            return {'out': out, 'x': h, 'x_1': torch.zeros_like(x),
                    'prop_mat': prop_mat, 'alpha_1': self.alpha_pos}

        # propagate_type: (x: Tensor)
        h = self.propagate(prop_mat, x=x)
        h -= self._multiply_coeff(x, self.beta)
        h -= self._multiply_coeff(x_1, alpha_1)
        h = self._devide_coeff(h, self.alpha_pos)
        out += self._forward_theta(h)

        return {
            'out': out,
            'x': h,
            'x_1': x,
            'prop_mat': prop_mat,
            'alpha_1': self.alpha_pos}

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(theta={self.theta})'
