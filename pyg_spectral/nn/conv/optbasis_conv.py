import torch
from torch import Tensor

from torch_geometric.typing import Adj
from pyg_spectral.nn.conv.base_mp import BaseMP


class OptBasisConv(BaseMP):
    r"""Convolutional layer with optimal adaptive basis.

    :paper: Graph Neural Networks with Learnable and Optimal Polynomial Bases
    :ref: https://github.com/yuziGuo/FarOptBasis/blob/master/layers/NormalBasisConv.py

    Args:
        num_hops, hop, cached: args for :class:`BaseMP`
    """
    name = lambda _: 'OptBasisConv'

    def __init__(self,
        num_hops: int = 0,
        hop: int = 0,
        cached: bool = True,
        **kwargs
    ):
        kwargs.setdefault('propagate_mat', 'A')
        super(OptBasisConv, self).__init__(num_hops, hop, cached, **kwargs)

    def _get_convolute_mat(self, x: Tensor, edge_index: Adj) -> dict:
        return {
            'x': x + torch.randn_like(x) * 1e-5,
            'x_1': torch.zeros_like(x),}

    def _forward(self,
        x: Tensor,
        x_1: Tensor,
        prop: Adj,
    ) -> dict:
        r"""
        Returns:
            x (Tensor): propagation result of :math:`k-1` (shape: :math:`(|\mathcal{V}|, F)`)
            x_1 (Tensor): propagation result of :math:`k-2` (shape: :math:`(|\mathcal{V}|, F)`)
            prop (Adj): propagation matrix
        """
        # dim_node = tuple(range(x.dim() - 1))
        dim_node = 0
        div_norm = lambda x: x / torch.clamp(torch.norm(x, dim=dim_node), min=1e-8)
        if self.hop == 0:
            h = div_norm(x)
            return {'x': h, 'x_1': torch.zeros_like(x), 'prop': prop}

        # propagate_type: (x: Tensor)
        h = self.propagate(prop, x=x)
        temp = torch.einsum('nh,nh->h', h, x)
        h = h - torch.einsum('h,nh->nh', temp, x)
        # FIXME: h is already not Ax here (require ortho to be equivalent?)
        temp = torch.einsum('nh,nh->h', h, x_1)
        h = h - torch.einsum('h,nh->nh', temp, x_1)
        h = div_norm(h)

        return {'x': h, 'x_1': x, 'prop': prop}
