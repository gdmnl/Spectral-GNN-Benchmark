import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.typing import Adj
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import reset

from pyg_spectral.nn.conv.base_mp import BaseMP


class ACMConv(BaseMP):
    r"""Convolutional layer of FBGNN & ACMGNN(I & II).

    :paper: Revisiting Heterophily For Graph Neural Networks
    :paper: Complete the Missing Half: Augmenting Aggregation Filtering with Diversification for Graph Convolutional Networks
    :ref: https://github.com/SitaoLuan/ACM-GNN/blob/main/ACM-Geometric/layers.py

    Args:
        alpha: variant I (propagate first) or II (act first)
        num_hops, hop, cached: args for :class:`BaseMP`
    """
    supports_batch: bool = False
    name = lambda args: f'ACMConv-{args.alpha:d}'
    pargs = ['alpha']

    def __init__(self,
        num_hops: int = 0,
        hop: int = 0,
        alpha: int = None,
        cached: bool = True,
        out_channels: int = None,
        **kwargs
    ):
        kwargs.setdefault('propagate_mat', 'A,L')
        super(ACMConv, self).__init__(num_hops, hop, cached, **kwargs)
        self.alpha = 1 if alpha is None else int(alpha)
        self.out_channels = out_channels

    def _init_with_theta(self):
        r"""
        Attributes:
            theta (torch.nn.ModuleDict): Linear transformation for each scheme.
        """
        self.schemes = self.theta.keys()
        self.n_scheme = len(self.schemes)

        self.norms = nn.ModuleDict({
            sch: nn.LayerNorm(self.out_channels) for sch in self.schemes})
        self.attentions_vec = nn.ModuleDict({
            sch: Linear(
                self.out_channels, 1,
                bias=False, weight_initializer='uniform') for sch in self.schemes})
        self.attentions_cat = Linear(
            self.n_scheme, self.n_scheme,
            bias=False, weight_initializer='uniform')

    def reset_parameters(self):
        for v in self.theta.values():
            reset(v)
        for v in self.norms.values():
            v.reset_parameters()
        for v in self.attentions_vec.values():
            v.reset_parameters()
        self.attentions_cat.reset_parameters()

    def _get_forward_mat(self, x: Tensor, edge_index: Adj) -> dict:
        return {'out': x}

    def _get_convolute_mat(self, x: Tensor, edge_index: Adj) -> dict:
        return {'out': x}

    def _forward_theta(self, x, scheme):
        r"""
        Attributes:
            theta (torch.nn.ModuleDict): Linear transformation for each scheme.
        """
        if callable(self.theta[scheme]):
            return self.theta[scheme](x)
        return self.theta[scheme] * x

    def forward(self,
        out: Tensor,
        prop_0: Adj,
        prop_1: Adj,
    ) -> dict:
        r"""
        Returns:
            out (Tensor): current propagation result (shape: :math:`(|\mathcal{V}|, F)`)
            prop_0, prop_1 (SparseTensor): propagation matrices
        """
        h, a = {}, {}
        prop_mat = {'low': prop_0, 'high': prop_1}
        for sch in self.schemes:
            h[sch] = self._forward_theta(x=out, scheme=sch)
            # propagate_type: (x: Tensor)
            if sch in ['low', 'high'] and self.alpha == 1:
                h[sch] = self.propagate(prop_mat[sch], x=h[sch])
            h[sch] = F.relu(h[sch])
            if sch in ['low', 'high'] and self.alpha == 2:
                h[sch] = self.propagate(prop_mat[sch], x=h[sch])

            a[sch] = self.norms[sch](h[sch])
            a[sch] = self.attentions_vec[sch](a[sch])

        a = torch.cat([a[sch] for sch in self.schemes], dim=1)
        a = self.attentions_cat(F.sigmoid(a)) / self.n_scheme
        a = F.softmax(a, dim=1)
        a = {sch: a[:, i][:, None] for i, sch in enumerate(self.schemes)}

        h = {sch: h[sch] * a[sch] for sch in self.schemes}
        out = torch.zeros_like(next(iter(h.values())))
        for sch in self.schemes:
            out += h[sch]
        out *= self.n_scheme

        return {'out': out, 'prop_0': prop_0, 'prop_1': prop_1,}

    def __repr__(self) -> str:
        if self.alpha == 1:
            return f'{self.__class__.__name__}(theta={self.theta})'
        else:
            return f'{self.__class__.__name__}II(theta={self.theta})'


# FEATURE: attr mapping in ACM++
