from typing import Optional, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.typing import Adj
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import spmm

from pyg_spectral.utils import get_laplacian


class ChebConv(MessagePassing):
    r"""Convolutional layer with Chebyshev Polynomials.
    paper: Convolutional Neural Networks on Graphs with Chebyshev Approximation, Revisited
    ref: https://github.com/yuziGuo/FarOptBasis/blob/master/models/FavardNormalNN.py

    Args:
        num_hops (int), hop (int): total and current number of propagation hops.
            hop=0 explicitly handles x without propagation.
        alpha (float): decay factor for each hop :math:`1/hop^\alpha`.
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
        alpha: float = -1.0,
        cached: bool = True,
        hidden_channels: int = 64,
        **kwargs
    ):
        kwargs.setdefault('aggr', 'add')
        super(ChebConv, self).__init__(**kwargs)

        self.num_hops = num_hops
        self.hop = hop
        self.theta = theta
        self.alpha = 0.0 if alpha < 0 else alpha  #NOTE: set actual alpha default here
        self.temps = None
        self.hidden_channels = hidden_channels

        self.cached = cached
        self._cache = None

    def reset_cache(self):
        del self._cache
        self._cache = None

    def reset_parameters(self):
        if hasattr(self.theta, 'reset_parameters'):
            self.theta.reset_parameters()
        
        t = torch.zeros(self.hidden_channels)
        t[0] = 1
        # t = t.repeat(self.n_channel, 1)
        self.alpha_params = nn.Parameter(t.float())
        self.yitas = nn.Parameter(torch.zeros(self.hidden_channels).float()) # (n_channels, )
        self.sqrt_betas = nn.Parameter(torch.zeros(self.hidden_channels).float()) # (n_channels, )

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
            # L_norm -> L_norm - I
            diag = edge_index.get_diag()
            edge_index = edge_index.set_diag(diag - 1.0)

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
        """
        return {
            'out': torch.zeros_like(x),
            'x': x,
            'x_1': x,
            'prop_mat': self.get_propagate_mat(x, edge_index),
            'yitas': self.yitas,
            'sqrt_betas': self.sqrt_betas}

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
        yitas: Tensor,
        sqrt_betas: Tensor,
    ) -> dict:
        r"""
        Args & Returns: (dct): same with output of get_forward_mat()
        """
        if self.hop == 0:
            return {'out': torch.zeros_like(x), 'x': x, 'x_1': torch.zeros_like(x), 'prop_mat': prop_mat, 'yitas': self.yitas, 'sqrt_betas': self.sqrt_betas}
        elif self.hop == 1:
            h = self._forward_theta(x) / self.sqrt_betas
            out += self.alpha_params * h
            return {'out': out, 'x': h, 'x_1': x_1, 'prop_mat': prop_mat, 'yitas': self.yitas, 'sqrt_betas': self.sqrt_betas}

        h = self.propagate(prop_mat, x=x)
        h = h - yitas * x -  sqrt_betas * x_1
        h = h / self.sqrt_betas 
        h = self._forward_theta(h)

        out += self.alpha_params *  h

        return {
            'out': out,
            'x': h,
            'x_1': x,
            'prop_mat': prop_mat,
            'yitas': self.yitas,
            'sqrt_betas': self.sqrt_betas}

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(alpha={self.alpha})'
