from typing import Optional, Any, Tuple
import re

import torch
from torch import Tensor

from torch_geometric.typing import Adj, SparseTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import spmm

from pyg_spectral.utils import get_laplacian


class BaseMP(MessagePassing):
    r"""Base filter layer structure.

    Args:
        num_hops (int): total number of propagation hops.
        hop (int): current number of propagation hops of this layer.
        alpha (float): additional scaling for self-loop in adjacency matrix
            :math:`\mathbf{A} + \alpha\mathbf{I}`, i.e. `improved` in PyG GCNConv.
        cached: whether cache the propagation matrix.
        **kwargs: Additional arguments of :class:`pyg.nn.conv.MessagePassing`.
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
        self.propagate_mat = kwargs.pop('propagate_mat', 'A')
        self.propagate_mat = self.propagate_mat.split(',')
        super(BaseMP, self).__init__(**kwargs)

        self.num_hops = num_hops
        self.hop = hop

        self.precomputed = False
        self.out_scale = 1.0
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
        Properties:
            self.propagate_mat (str): propagation schemes, separated by ','.
                Each scheme starts with 'A' or 'L' for adjacency or Laplacian.
                Can follow '+[p]*I' or '-[p]*I' for adjusting diagonal, where
                `p` can be float or attribute name.
        Returns:
            prop (SparseTensor): propagation matrix
        """
        cache = self._cache
        if cache is None:
            mats = self._get_propagate_mat(x, edge_index)
            if self.cached:
                self._cache = mats
        else:
            mats = cache
        return mats

    def _get_propagate_mat(self,
        x: Tensor,
        edge_index: Adj
    ) -> Adj:
        """ Shadow function for self.get_propagate_mat().
            edge_index (SparseTensor or torch.sparse_csr_tensor)
        """
        def _get_adj(mat: Adj, diag: float):
            if diag != 0:
                if isinstance(mat, SparseTensor):
                    dg = mat.get_diag()
                    return mat.set_diag(dg + diag)
                else:
                    diag = torch.ones(mat.size(0)) * diag
                    dg = torch.sparse.spdiags(diag, torch.tensor(0), mat.size(),
                                              layout=torch.sparse_csr)
                    dg = dg.to(mat.device, mat.dtype)
                    return mat + dg
            return mat

        def _get_lap(mat: Adj, diag: float):
            return get_laplacian(
                mat,
                normalization=True,
                diag=1.0+diag,
                dtype=mat.dtype)

        pattern = re.compile(r'([AL])([\+\-]([\d\w\._]*)\*?I)?')
        mats = {}
        for i, scheme in enumerate(self.propagate_mat):
            match = pattern.match(scheme.strip())
            mati, diag_part, diag_value = match.groups()

            if diag_part is not None:
                if diag_value == '':
                    diag = float(diag_part[0]+'1')
                else:
                    try:
                        diag = float(diag_part[0]+diag_value)
                    except ValueError:
                        diag = getattr(self, diag_value)
            else:
                diag = 0.0

            if mati == 'A':
                mats[f'prop_{i}'] = _get_adj(edge_index, diag)
            else:
                mats[f'prop_{i}'] = _get_lap(edge_index, diag)

        if len(mats) == 1:
            return mats['prop_0']
        return mats

    def get_forward_mat(self,
        x: Tensor,
        edge_index: Adj,
    ) -> dict:
        r"""Get matrices for self.forward(). Called during forward().

        Args:
            x (Tensor), edge_index (Adj): from pyg.data.Data
        Returns:
            out (:math:`(|\mathcal{V}|, F)` Tensor): output tensor
            prop (Adj): propagation matrix
        """
        raise NotImplementedError

    def _forward_theta(self, x):
        r"""
        theta (nn.Parameter or nn.Module): transformation of propagation result
            before applying to the output.
        """
        if callable(self.theta):
            return self.theta(x)
        else:
            return self.theta * x

    def _forward_out(self, out: Tensor, conv_out: dict) -> Tensor:
        r"""Default to use the first tensor for calculating output."""
        x = next(iter(conv_out.values()))
        if self.out_scale == 1:
            out += self._forward_theta(x)
        else:
            out += self._forward_theta(x) * self.out_scale
        return out

    def forward(self, **kwargs) -> dict:
        r""" Wrapper for distinguishing precomputed outputs.
        Args & Returns (dct): same with output of get_forward_mat()
        """
        if self.precomputed:
            return self._forward_out(**kwargs)
        else:
            out = kwargs.pop('out')
            conv_out, rest_out = self._forward(**kwargs)
            out = self._forward_out(out, conv_out)
            return {'out': out, **conv_out, **rest_out}

    def _forward(self,
        x: Tensor,
        prop: Adj,
    ) -> Tuple[dict, dict]:
        r""" Shadow function for self.forward() to be implemented in subclasses
            without calculating output.
            if `self.supports_batch == True`, then should not contain derivable computations.
        Returns:
            conv_out (dict): tensors for calculating output
            rest_out (dict): remained tensors for next layer
        """
        raise NotImplementedError

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        # return spmm(adj_t, x, reduce=self.aggr)   # torch_sparse.SparseTensor
        return torch.spmm(adj_t, x)                 # torch.sparse.Tensor

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(theta={self.theta})'
