import torch
from torch import Tensor

from torch_geometric.typing import Adj
from pyg_spectral.nn.conv.base_mp import BaseMP


class OptBasisConv(BaseMP):
    r"""Convolutional layer with optimal adaptive basis.
    paper: Graph Neural Networks with Learnable and Optimal Polynomial Bases
    ref: https://github.com/yuziGuo/FarOptBasis/blob/master/layers/NormalBasisConv.py

    Args:
        --- BaseMP Args ---
        num_hops (int), hop (int): total and current number of propagation hops.
        cached: whether cache the propagation matrix.
    """
    def __init__(self,
        num_hops: int = 0,
        hop: int = 0,
        cached: bool = True,
        **kwargs
    ):
        kwargs.setdefault('propagate_mat', 'A')
        super(OptBasisConv, self).__init__(num_hops, hop, cached, **kwargs)

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
            prop (Adj): propagation matrix
        """
        return {
            'out': torch.zeros_like(x),
            'x': x + torch.randn_like(x) * 1e-5,
            'x_1': torch.zeros_like(x),
            'prop': self.get_propagate_mat(x, edge_index)}

    def forward(self,
        out: Tensor,
        x: Tensor,
        x_1: Tensor,
        prop: Adj,
    ) -> dict:
        r"""
        Args & Returns: (dct): same with output of get_forward_mat()
        """
        # dim_node = tuple(range(x.dim() - 1))
        dim_node = 0
        div_norm = lambda x: x / torch.clamp(torch.norm(x, dim=dim_node), min=1e-8)
        if self.hop == 0:
            h = div_norm(x)
            out = self._forward_theta(h)
            return {'out': out, 'x': h, 'x_1': torch.zeros_like(x),
                    'prop': prop}

        # propagate_type: (x: Tensor)
        h = self.propagate(prop, x=x)
        temp = torch.einsum('nh,nh->h', h, x)
        h = h - torch.einsum('h,nh->nh', temp, x)
        # FIXME: h is already not Ax here (require ortho to be equivalent?)
        temp = torch.einsum('nh,nh->h', h, x_1)
        h = h - torch.einsum('h,nh->nh', temp, x_1)
        h = div_norm(h)
        out += self._forward_theta(h)

        return {
            'out': out,
            'x': h,
            'x_1': x,
            'prop': prop}
