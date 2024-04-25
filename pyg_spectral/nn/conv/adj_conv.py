from typing import Optional, Any, Union

import torch
import torch.nn as nn
from torch import Tensor

from torch_geometric.typing import Adj
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import spmm


class AdjConv(MessagePassing):
    r"""Linear filter using the normalized adjacency matrix for propagation.

    Args:
        num_hops (int), hop (int): total and current number of propagation hops.
        alpha (float): additional scaling for self-loop in adjacency matrix
            :math:`\mathbf{A} + \alpha\mathbf{I}`.
        theta (nn.Parameter or nn.Module): transformation of propagation result
            before applying to the output.
        cached: whether cache the propagation matrix.
    """
    supports_batch: bool = False
    supports_norm_batch: bool = False
    _cache: Optional[Any]

    def __init__(self,
        num_hops: int = 0,
        hop: int = 0,
        theta: Union[nn.Parameter, nn.Module] = None,
        alpha: float = 0,
        cached: bool = True,
        **kwargs
    ):
        # FEATURE: `combine_root` as `pyg.nn.conv.SimpleConv`
        kwargs.setdefault('aggr', 'add')
        super(AdjConv, self).__init__(**kwargs)

        self.num_hops = num_hops
        self.hop = hop
        self.theta = theta
        self.alpha = alpha

        self.cached = cached
        self._cache = None

    def reset_cache(self):
        del self._cache
        self._cache = None

    def reset_parameters(self):
        if hasattr(self.theta, 'reset_parameters'):
            self.theta.reset_parameters()
        self.reset_cache()

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
            # A_norm -> A_norm + alpha * I
            if self.alpha != 0:
                diag = edge_index.get_diag()
                edge_index = edge_index.set_diag(diag + self.alpha)

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
            x (:math:`(|\mathcal{V}|, F)` Tensor): current propagation result
            prop_mat (Adj): propagation matrix
        """
        return {
            'out': torch.zeros_like(x),
            'x': x,
            'prop_mat': self.get_propagate_mat(x, edge_index)}

    def _forward_theta(self, x):
        if callable(self.theta):
            return self.theta(x)
        else:
            return self.theta * x

    def forward(self,
        out: Tensor,
        x: Tensor,
        prop_mat: Adj,
    ) -> dict:
        r"""
        Args & Returns: (dct): same with output of get_forward_mat()
        """
        if self.hop == 0 and not callable(self.theta):
            # No transformation
            out = self._forward_theta(x)
            return {'out': out, 'x': x, 'prop_mat': prop_mat}

        # propagate_type: (x: Tensor)
        x = self.propagate(prop_mat, x=x)

        # NOTE: different to GCN, here conduct propagation first, then transformation
        out += self._forward_theta(x)

        return {
            'out': out,
            'x': x,
            'prop_mat': prop_mat}

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(theta={self.theta})'
