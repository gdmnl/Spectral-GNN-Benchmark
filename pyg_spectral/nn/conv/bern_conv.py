from typing import Optional, Any

from scipy.special import comb
import torch
from torch import Tensor
import torch.nn.functional as F

from torch_geometric.typing import Adj
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import spmm

from pyg_spectral.utils import get_laplacian


class BernConv(MessagePassing):
    r"""Convolutional layer with Bernstein Polynomials.
    We propose a new implementation reducing memory from O(KFn) to O(3Fn).
    paper: BernNet: Learning Arbitrary Graph Spectral Filters via Bernstein Approximation
    ref: https://github.com/ivam-he/BernNet/blob/main/NodeClassification/Bernpro.py

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
        super(BernConv, self).__init__(**kwargs)

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

    def get_propagate_mat(self,
        x: Tensor,
        edge_index: Adj
    ) -> dict:
        r"""Get matrices for self.propagate(). Called before each forward() with
            same input.

        Args:
            x (Tensor), edge_index (Adj): from pyg.data.Data
        Returns:
            prop_mat_1 (SparseTensor): L
            prop_mat_2 (SparseTensor): 2I - L
        """
        cache = self._cache
        if cache is None:
            # 2I - L_norm = I + A_norm
            diag2 = edge_index.get_diag()
            mat2 = edge_index.set_diag(diag2 + 1.0)

            # A_norm -> L_norm
            mat1 = get_laplacian(
                edge_index,
                normalization=True,
                dtype=x.dtype)

            res = {
                'prop_mat_1': mat1,
                'prop_mat_2': mat2}
            if self.cached:
                self._cache = res
        else:
            res = cache
        return res

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
            x (:math:`(|\mathcal{V}|, F)` Tensor): propagation result through (2I-L)
            prop_mat_1 (SparseTensor): L
            prop_mat_2 (SparseTensor): 2I - L
        """
        return {
            'out': torch.zeros_like(x),
            'x': x
            } | self.get_propagate_mat(x, edge_index)

    def _forward_theta(self, x):
        if callable(self.theta):
            return self.theta(x)
        else:
            return self.theta * F.relu(x)

    def forward(self,
        out: Tensor,
        x: Tensor,
        prop_mat_1: Adj,
        prop_mat_2: Adj,
    ) -> dict:
        r"""
        Args & Returns: (dct): same with output of get_forward_mat()
        """
        if self.hop > 0:
            # propagate_type: (x: Tensor)
            x = self.propagate(prop_mat_2, x=x)

        h = x
        # Propagate through L for (K-k) times
        for _ in range(self.num_hops - self.hop):
            # propagate_type: (x: Tensor)
            h = self.propagate(prop_mat_1, x=h)

        h = self._forward_theta(h)
        out += h * comb(self.num_hops, self.num_hops-self.hop) / (2**self.num_hops)

        return {
            'out': out,
            'x': x,
            'prop_mat_1': prop_mat_1,
            'prop_mat_2': prop_mat_2}

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(theta={self.theta})'
