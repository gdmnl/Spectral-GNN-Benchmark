from typing import Optional

import torch
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from pyg_spectral.nn.models.decoupled_compose import DecoupledFixedCompose, DecoupledVarCompose


class PrecomputedFixedCompose(DecoupledFixedCompose):
    def convolute(self,
        x: Tensor,
        edge_index: Adj,
        batch: OptTensor = None,
        batch_size: Optional[int] = None,
    ) -> Tensor:
        r"""Decoupled propagation step for calling the convolutional module.
            Requires no variable transformation in conv.forward().
        Returns:
            embed (Tensor): Precomputed node embeddings.
                Shape: :math:`(|\mathcal{V}|, F, Q)`.
        """
        out = []
        conv_mats = self.get_forward_mat()
        for i, channel in enumerate(self.convs):
            conv_mat = conv_mats[i](x, edge_index)
            for conv in channel:
                conv_mat = conv(**conv_mat)
            out.append(conv_mat['out'])
        return torch.stack(out, dim=-1)

    def forward(self,
        x: Tensor,
        batch: OptTensor = None,
        batch_size: Optional[int] = None,
    ) -> Tensor:
        r"""
        Args:
            x (Tensor): the output `embed` from `convolute()`.
            batch (Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example.
                Only needs to be passed in case the underlying normalization
                layers require the :obj:`batch` information.
                (default: :obj:`None`)
            batch_size (int, optional): The number of examples :math:`B`.
                Automatically calculated if not given.
                Only needs to be passed in case the underlying normalization
                layers require the :obj:`batch` information.
                (default: :obj:`None`)
        """
        out = None
        for i, channel in enumerate(self.convs):
            if i == 0:
                out = x[..., i]
            else:
                if self.combine == 'sum':
                    out = out + x[..., i]
                elif self.combine in ['sum_weighted', 'sum_vec']:
                    out = out + self.gamma[i] * x[..., i]
                elif self.combine == 'cat':
                    out = torch.cat((out, x[..., i]), dim=-1)

        if self.out_layers > 0:
            out = self.out_mlp(out, batch=batch, batch_size=batch_size)
        return out


class PrecomputedVarCompose(DecoupledVarCompose):
    def convolute(self,
        x: Tensor,
        edge_index: Adj,
        batch: OptTensor = None,
        batch_size: Optional[int] = None,
    ) -> Tensor:
        r"""Decoupled propagation step for calling the convolutional module.
            Requires no variable transformation in conv.forward().
        Returns:
            embed (Tensor): List of precomputed node embeddings of each hop.
                Shape: :math:`(|\mathcal{V}|, F, Q, len(convs)+1)`.
        """
        out = []
        conv_mats = self.get_forward_mat()
        for i, channel in enumerate(self.convs):
            conv_mat = conv_mats[i](x, edge_index, comp_scheme='convolute')

            xi = conv_mat['x'] if 'x' in conv_mat else conv_mat['out']
            xs = [xi]
            for conv in channel:
                conv.comp_scheme = 'convolute'
                conv_mat = conv._forward(**conv_mat)
                xi = conv_mat['x'] if 'x' in conv_mat else conv_mat['out']
                xs.append(xi)

            out.append(torch.stack(xs, dim=xs[0].dim()))
        return torch.stack(out, dim=-2)

    def forward(self,
        xs: Tensor,
        batch: OptTensor = None,
        batch_size: Optional[int] = None,
    ) -> Tensor:
        r"""
        Args:
            x (Tensor): the output `embed` from `convolute()`.
            batch (Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example.
                Only needs to be passed in case the underlying normalization
                layers require the :obj:`batch` information.
                (default: :obj:`None`)
            batch_size (int, optional): The number of examples :math:`B`.
                Automatically calculated if not given.
                Only needs to be passed in case the underlying normalization
                layers require the :obj:`batch` information.
                (default: :obj:`None`)
        """
        out = None
        conv_mats = self.get_forward_mat()

        for i, channel in enumerate(self.convs):
            conv_mat = conv_mats[i](xs[..., i, 0], None, comp_scheme='forward')
            for k, conv in enumerate(channel):
                conv.comp_scheme = 'forward'
                key = 'x' if 'x' in conv._forward.__code__.co_varnames else 'out'
                conv_mat[key] = xs[..., i, k+1]
                conv_mat = conv(**conv_mat)

            if i == 0:
                out = conv_mat['out']
            else:
                if self.combine == 'sum':
                    out = out + conv_mat['out']
                elif self.combine in ['sum_weighted', 'sum_vec']:
                    out = out + self.gamma[i] * conv_mat['out']
                elif self.combine == 'cat':
                    out = torch.cat((out, conv_mat['out']), dim=-1)

        if self.out_layers > 0:
            out = self.out_mlp(out, batch=batch, batch_size=batch_size)
        return out
