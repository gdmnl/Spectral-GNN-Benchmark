from typing import Optional, Any

from torch import Tensor

from torch_geometric.typing import Adj
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import spmm

from pyg_spectral.utils import get_laplacian


class LapiConv(MessagePassing):
    r"""Iterative linear filter using the normalized adjacency matrix.
    Used in AdaGNN.
    paper: AdaGNN: Graph Neural Networks with Adaptive Frequency Response Filter
    ref: https://github.com/yushundong/AdaGNN/blob/main/layers.py

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
        super(LapiConv, self).__init__(**kwargs)

        self.num_hops = num_hops
        self.hop = hop

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
            # A_norm -> L_norm
            edge_index = get_laplacian(
                edge_index,
                normalization=True,
                dtype=x.dtype)

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
            out (:math:`(|\mathcal{V}|, F)` Tensor): current propagation result
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
        # propagate_type: (x: Tensor)
        h = self.propagate(prop_mat, x=out)
        h = out - self._forward_theta(h)

        return {
            'out': h,
            'prop_mat': prop_mat}

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(theta={self.theta})'
