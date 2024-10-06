import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.typing import Adj
from torch_geometric.nn.inits import reset
from pyg_spectral.nn.conv.base_mp import BaseMP


class FavardConv(BaseMP):
    r"""Convolutional layer with basis in Favard's Theorem.

    :paper: Graph Neural Networks with Learnable and Optimal Polynomial Bases
    :ref: https://github.com/yuziGuo/FarOptBasis/blob/master/layers/FavardNormalConv.py

    Args:
        num_hops, hop, cached: args for :class:`BaseMP`
    """
    supports_batch: bool = False
    name = lambda _: 'FavardConv'

    def __init__(self,
        num_hops: int = 0,
        hop: int = 0,
        cached: bool = True,
        **kwargs
    ):
        kwargs.setdefault('propagate_mat', 'A')
        super(FavardConv, self).__init__(num_hops, hop, cached, **kwargs)

    def _init_with_theta(self):
        if callable(self.theta):
            self.alpha = copy.deepcopy(self.theta)
            self.beta  = copy.deepcopy(self.theta)
        else:
            self.register_parameter('alpha', nn.Parameter(self.theta.data.clone()))
            self.register_parameter('beta' , nn.Parameter(self.theta.data.clone()))

    def reset_parameters(self):
        if hasattr(self.theta, 'reset_parameters'):
            reset(self.theta)
            reset(self.alpha)
            reset(self.beta)
        else:
            self.alpha.data = self.theta.data.clone()
            self.beta.data = self.theta.data.clone()

    def _get_convolute_mat(self, x: Tensor, edge_index: Adj) -> dict:
        return {
            'x': x + torch.randn_like(x) * 1e-5,
            'x_1': torch.zeros_like(x),}

    def _get_forward_mat(self, x: Tensor, edge_index: Adj) -> dict:
        r"""
        Returns:
            out (Tensor): initial output tensor (shape: :math:`(|\mathcal{V}|, F)`)
            alpha_1: parameter for :math:`k-1`
        """
        return {'out': torch.zeros_like(x), 'alpha_1': None}

    @staticmethod
    def _mul_coeff(x, coeff, pos=True):
        if callable(coeff):
            return F.relu(coeff(x))
        else:
            coeff = torch.clamp(coeff, min=1e-2) if pos else coeff
            return coeff * x

    @staticmethod
    def _div_coeff(x, coeff, pos=True):
        if callable(coeff):
            assert hasattr(coeff, 'weight')
            # use L2 norm of weight
            return x / torch.norm(coeff.weight)
        else:
            coeff = torch.clamp(coeff, min=1e-2) if pos else coeff
            return x / coeff

    def _forward(self,
        x: Tensor,
        x_1: Tensor,
        prop: Adj,
        alpha_1: nn.Parameter | nn.Module
    ) -> dict:
        r"""
        Returns:
            x (Tensor): propagation result of :math:`k-1` (shape: :math:`(|\mathcal{V}|, F)`)
            x_1 (Tensor): propagation result of :math:`k-2` (shape: :math:`(|\mathcal{V}|, F)`)
            prop (Adj): propagation matrix
            alpha_1: parameter for :math:`k-1`
        """
        if self.hop == 0:
            h = self._div_coeff(x, self.alpha, pos=True)
            return {'x': h, 'x_1': torch.zeros_like(x),
                    'prop': prop, 'alpha_1': self.alpha}

        # propagate_type: (x: Tensor)
        h = self.propagate(prop, x=x)
        h -= self._mul_coeff(x, self.beta, pos=False)
        h -= self._mul_coeff(x_1, alpha_1, pos=True)
        h = self._div_coeff(h, self.alpha, pos=True)

        return {'x': h, 'x_1': x, 'prop': prop, 'alpha_1': self.alpha}
