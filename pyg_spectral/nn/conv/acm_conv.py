from typing import Optional, Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.typing import Adj
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import spmm

from pyg_spectral.utils import get_laplacian


class ACMConv(MessagePassing):
    r"""Convolutional layer of FBGNN & ACMGNN(I & II).
    paper: Revisiting Heterophily For Graph Neural Networks
    paper: Complete the Missing Half: Augmenting Aggregation Filtering with Diversification for Graph Convolutional Networks
    ref: https://github.com/SitaoLuan/ACM-GNN/blob/main/ACM-Geometric/layers.py

    Args:
        num_hops (int), hop (int): total and current number of propagation hops.
            hop=0 explicitly handles x without propagation.
        alpha (int): variant I (propagate first) or II (act first)
        theta (nn.ModuleDict): transformation of propagation result
            before applying to the output.
        cached: whether cache the propagation matrix.
    """
    supports_batch: bool = False
    supports_norm_batch: bool = False
    _cache: Optional[Any]

    def __init__(self,
        num_hops: int = 0,
        hop: int = 0,
        theta: nn.ModuleDict = None,
        alpha: int = 1,
        cached: bool = True,
        **kwargs
    ):
        kwargs.setdefault('aggr', 'add')
        super(ACMConv, self).__init__(**kwargs)

        self.num_hops = num_hops
        self.hop = hop
        self.theta = theta
        self.schemes = theta.keys()
        self.n_scheme = len(self.schemes)
        self.alpha = 1 if alpha <= 0 else int(alpha)  #NOTE: set actual alpha default here

        self.cached = cached
        self._cache = None

        self.norms = nn.ModuleDict({
            sch: nn.LayerNorm(self.theta[sch].out_channels) for sch in self.schemes})
        self.attentions_vec = nn.ModuleDict({
            sch: Linear(
                self.theta[sch].out_channels, 1,
                bias=False, weight_initializer='uniform') for sch in self.schemes})
        self.attentions_cat = Linear(
            self.n_scheme, self.n_scheme,
            bias=False, weight_initializer='uniform')

    def reset_cache(self):
        del self._cache
        self._cache = None

    def reset_parameters(self):
        for v in self.theta.values():
            v.reset_parameters()
        for v in self.norms.values():
            v.reset_parameters()
        for v in self.attentions_vec.values():
            v.reset_parameters()
        self.attentions_cat.reset_parameters()

    def get_propagate_mat(self,
        x: Tensor,
        edge_index: Adj
    ) -> Dict[str, Adj]:
        r"""Get matrices for self.propagate(). Called before each forward() with
            same input.

        Args:
            x (Tensor), edge_index (Adj): from pyg.data.Data
        Returns:
            prop_mat_low, prop_mat_high (SparseTensor): propagation matrices
        """
        cache = self._cache
        if cache is None:
            prop_mat_low = edge_index
            # A_norm -> L_norm
            prop_mat_high = get_laplacian(
                edge_index.clone(),
                normalization=True,
                dtype=x.dtype)

            if self.cached:
                dct = {'prop_mat_low': prop_mat_low, 'prop_mat_high': prop_mat_high}
                self._cache = dct
        else:
            dct = cache
        return dct

    def get_forward_mat(self,
        x: Tensor,
        edge_index: Adj,
    ) -> dict:
        r"""Get matrices for self.forward(). Called during forward().

        Args:
            x (Tensor), edge_index (Adj): from pyg.data.Data
        Returns:
            out (:math:`(|\mathcal{V}|, F)` Tensor): current propagation result
            prop_mat_low, prop_mat_high (SparseTensor): propagation matrices
        """
        return {'out': x} | self.get_propagate_mat(x, edge_index)

    def _forward_theta(self, x, scheme):
        return self.theta[scheme](x)

    def forward(self,
        out: Tensor,
        prop_mat_low: Adj,
        prop_mat_high: Adj,
    ) -> dict:
        r"""
        Args & Returns: (dct): same with output of get_forward_mat()
        """
        h, a = {}, {}
        prop_mat = {'low': prop_mat_low, 'high': prop_mat_high}
        for sch in self.schemes:
            h[sch] = self._forward_theta(out, scheme=sch)
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

        return {
            'out': out,
            'prop_mat_low': prop_mat_low,
            'prop_mat_high': prop_mat_high,}

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        if self.alpha == 1:
            return f'{self.__class__.__name__}(theta={self.theta})'
        else:
            return f'{self.__class__.__name__}II(theta={self.theta})'


# FEATURE: attr mapping in ACM++
