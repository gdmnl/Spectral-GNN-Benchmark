from typing import Optional, Any, Union

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.typing import Adj
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.utils import spmm


class FavardConv(MessagePassing):
    r"""Convolutional layer with basis in Favard's Theorem.
    paper: Graph Neural Networks with Learnable and Optimal Polynomial Bases
    ref: https://github.com/yuziGuo/FarOptBasis/blob/master/layers/FavardNormalConv.py

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

    def _init_with_theta(self):
        if callable(self.theta):
            self.alpha = copy.deepcopy(self.theta)
            self.beta  = copy.deepcopy(self.theta)
        else:
            self.register_parameter('alpha', nn.Parameter(self.theta.data.clone()))
            self.register_parameter('beta' , nn.Parameter(self.theta.data.clone()))

    def reset_cache(self):
        del self._cache
        self._cache = None

    def reset_parameters(self):
        if hasattr(self.theta, 'reset_parameters'):
            reset(self.theta)
            reset(self.alpha)
            reset(self.beta)
        else:
            self.alpha.data = self.theta.data.clone()
            self.beta.data = self.theta.data.clone()

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
            'x': x + torch.randn_like(x) * 1e-5,
            'x_1': torch.zeros_like(x),
            'prop_mat': self.get_propagate_mat(x, edge_index),
            'alpha_1': None}

    def _mul_coeff(self, x, coeff, pos=True):
        if callable(coeff):
            return F.relu(coeff(x))
        else:
            coeff = torch.clamp(coeff, min=1e-2) if pos else coeff
            return coeff * x

    def _div_coeff(self, x, coeff, pos=True):
        if callable(coeff):
            assert hasattr(coeff, 'weight')
            # use L2 norm of weight
            return x / torch.norm(coeff.weight)
        else:
            coeff = torch.clamp(coeff, min=1e-2) if pos else coeff
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
            h = self._div_coeff(x, self.alpha, pos=True)
            out = self._forward_theta(h)
            return {'out': out, 'x': h, 'x_1': torch.zeros_like(x),
                    'prop_mat': prop_mat, 'alpha_1': self.alpha}

        # propagate_type: (x: Tensor)
        h = self.propagate(prop_mat, x=x)
        h -= self._mul_coeff(x, self.beta, pos=False)
        h -= self._mul_coeff(x_1, alpha_1, pos=True)
        h = self._div_coeff(h, self.alpha, pos=True)
        out += self._forward_theta(h)

        return {
            'out': out,
            'x': h,
            'x_1': x,
            'prop_mat': prop_mat,
            'alpha_1': self.alpha}

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(theta={self.theta})'
