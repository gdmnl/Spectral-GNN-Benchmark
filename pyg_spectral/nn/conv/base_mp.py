from typing import Optional, Any
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
        num_hops: total number of propagation hops.
        hop: current number of propagation hops of this layer.
        cached: whether cache the propagation matrix.
        **kwargs: Additional arguments of :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    supports_batch: bool = True
    supports_norm_batch: bool = True
    _cache: Optional[Any]

    def __init__(self,
        num_hops: int = 0,
        hop: int = 0,
        cached: bool = True,
        **kwargs
    ):
        kwargs.setdefault('aggr', 'add')
        self.propagate_be = kwargs.pop('propagate_be', 'torch.sparse')
        self.propagate_mat = kwargs.pop('propagate_mat', 'A')
        super(BaseMP, self).__init__(**kwargs)

        self.num_hops = num_hops
        self.hop = hop

        self.comp_scheme = None
        self.out_scale = 1.0
        self.cached = cached
        self._cache = None

    def reset_cache(self):
        del self._cache
        self._cache = None

    # ==========
    def get_propagate_mat(self,
        x: Tensor,
        edge_index: Adj
    ) -> Adj:
        r"""Get matrices for :meth:`propagate()`. Called before each
        :meth:`forward()` with same input.

        Args:
            x: from :class:`torch_geometric.data.Data`
            edge_index: from :class:`torch_geometric.data.Data`
        Attributes:
            propagate_mat (str): propagation schemes, separated by ``,``.
                Each scheme starts with ``A`` or ``L`` for adjacency or Laplacian,
                optionally following ``+[p*]I`` or ``-[p*]I`` for scaling the
                diagonal, where ``p`` can be float or attribute name.
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
        """ Shadow function for :meth:`get_propagate_mat()`.
        """
        def _get_adj(mat: Adj, diag: float):
            if diag != 0:
                if self.propagate_be == 'torch_sparse':
                    assert isinstance(mat, SparseTensor)
                    dg = mat.get_diag()
                    return mat.set_diag(dg + diag)
                elif self.propagate_be == 'torch.sparse':
                    diag = torch.ones(mat.size(0)) * diag
                    dg = torch.sparse.spdiags(diag, torch.tensor(0), mat.size(),
                                              layout=torch.sparse_csr)
                    dg = dg.to(mat.device, mat.dtype)
                    return mat + dg
                else:
                    raise NotImplementedError(f'Backend {self.propagate_be} not implemented.')
            return mat

        def _get_lap(mat: Adj, diag: float):
            return get_laplacian(
                mat,
                normalization=True,
                diag=1.0+diag,
                dtype=mat.dtype)

        pattern = re.compile(r'([AL])([\+\-]([\d\w\._]*)\*?I)?')
        mats = {}
        for i, scheme in enumerate(self.propagate_mat.split(',')):
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
            mats['prop'] = mats.pop('prop_0')
        return mats

    def _get_forward_mat(self, x: Tensor, edge_index: Adj) -> dict:
        r"""Returns should match the arg list of :meth:`forward()` when
        ``self.comp_scheme == 'forward'``.

        Returns:
            out (Tensor): initial output tensor (shape: :math:`(|\mathcal{V}|, F)`)
        """
        return {'out': torch.zeros_like(x),}

    def _get_convolute_mat(self, x: Tensor, edge_index: Adj) -> dict:
        r"""
        Returns should match the arg list of :meth:`forward()`.
        """
        return {'x': x,}

    def get_forward_mat(self,
        x: Tensor,
        edge_index: Adj,
        comp_scheme: Optional[str] = None
    ) -> dict:
        r"""Get matrices for :meth:`forward()`. Called during :meth:`forward()`.

        Args:
            x: from :class:`torch_geometric.data.Data`
            edge_index: from :class:`torch_geometric.data.Data`
        Returns:
            out (Tensor): output tensor (shape: :math:`(|\mathcal{V}|, F)`)
            prop (Adj): propagation matrix
        """
        comp_scheme = comp_scheme or self.comp_scheme
        if comp_scheme is None:
            return self._get_forward_mat(x, edge_index) \
                | self._get_convolute_mat(x, edge_index) \
                | self.get_propagate_mat(x, edge_index)
        if comp_scheme == 'forward':
            return self._get_forward_mat(x, edge_index)
        else:
            return self._get_convolute_mat(x, edge_index) \
                | self.get_propagate_mat(x, edge_index)

    # ==========
    def _forward_theta(self, **kwargs):
        r"""
        Attributes:
            theta (nn.Parameter | nn.Module): transformation of propagation
                result before applying to the output.
        """
        x = kwargs['x'] if 'x' in kwargs else kwargs['out']
        if callable(self.theta):
            return self.theta(x)
        else:
            return self.theta * x

    def _forward_out(self, **kwargs) -> Tensor:
        r""" Shadow function for calling :meth:`_forward_theta()` and accumulating results.

        Returns:
            out (Tensor): output tensor for accumulating propagation results
                (shape: :math:`(|\mathcal{V}|, F)`)
        """
        if self.out_scale == 1:
            res = self._forward_theta(**kwargs)
        else:
            res = self._forward_theta(**kwargs) * self.out_scale
        return kwargs['out'] + res

    def forward(self, **kwargs) -> dict:
        r""" Wrapper for distinguishing precomputed outputs.
        Args & Returns should match the output of :meth:`get_forward_mat()`
        """
        if self.comp_scheme is None or self.comp_scheme == 'convolute':
            fwd_kwargs, keys = {}, list(kwargs.keys())
            for k in keys:
                if k not in self._forward.__code__.co_varnames:
                    fwd_kwargs[k] = kwargs.pop(k)
            kwargs = self._forward(**kwargs) | fwd_kwargs
        kwargs['out'] = self._forward_out(**kwargs)
        return kwargs

    def _forward(self,
        x: Tensor,
        prop: Adj,
    ) -> dict:
        r"""Shadow function for :meth:`forward()` to be implemented in subclasses
        without calculating output.
        if ``self.supports_batch == True``, then should not contain derivable computations.
        Dicts of Args & Returns should be matched.

        Returns:
            x (Tensor): tensor for calculating :obj:`out`
        """
        raise NotImplementedError

    # ==========
    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        if self.propagate_be == 'torch_sparse':
            return spmm(adj_t, x, reduce=self.aggr)     # torch_sparse.SparseTensor
        elif self.propagate_be == 'torch.sparse':
            return torch.spmm(adj_t, x)                 # torch.sparse.Tensor
        else:
            raise NotImplementedError(f'Backend {self.propagate_be} not implemented.')

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(theta={self.theta})'
