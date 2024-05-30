from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from pyg_spectral.nn.models.decoupled import DecoupledFixed, DecoupledVar


class PrecomputedFixed(DecoupledFixed):
    # TODO: docstring
    def __init__(self,
            conv: str,
            num_hops: int = 0,
            in_channels: Optional[int] = None,
            hidden_channels: Optional[int] = None,
            out_channels: Optional[int] = None,
            in_layers: Optional[int] = None,
            out_layers: Optional[int] = None,
            dropout_lin: Union[float, List[float]] = 0.,
            dropout_conv: float = 0.,
            act: Union[str, Callable, None] = "relu",
            act_first: bool = False,
            act_kwargs: Optional[Dict[str, Any]] = None,
            norm: Union[str, Callable, None] = "batch_norm",
            norm_kwargs: Optional[Dict[str, Any]] = None,
            plain_last: bool = False,
            bias: Union[bool, List[bool]] = True,
            **kwargs):
        assert in_layers is None or in_layers == 0, "PrecomputedFixed does not support in_layers."
        super(PrecomputedFixed, self).__init__(
            conv=conv,
            num_hops=num_hops,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            in_layers=in_layers,
            out_layers=out_layers,
            dropout_lin=dropout_lin,
            dropout_conv=dropout_conv,
            act=act,
            act_first=act_first,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
            plain_last=plain_last,
            bias=bias,
            **kwargs
        )

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
        """
        conv_mat = self.get_forward_mat(x, edge_index)
        for conv in self.convs:
            conv_mat = conv(**conv_mat)
        return conv_mat['out']

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
        if self.out_layers > 0:
            x = self.out_mlp(x, batch=batch, batch_size=batch_size)
        return x


class PrecomputedVar(DecoupledVar):
    def __init__(self,
            conv: str,
            num_hops: int = 0,
            in_channels: Optional[int] = None,
            hidden_channels: Optional[int] = None,
            out_channels: Optional[int] = None,
            in_layers: Optional[int] = None,
            out_layers: Optional[int] = None,
            dropout_lin: Union[float, List[float]] = 0.,
            dropout_conv: float = 0.,
            act: Union[str, Callable, None] = "relu",
            act_first: bool = False,
            act_kwargs: Optional[Dict[str, Any]] = None,
            norm: Union[str, Callable, None] = "batch_norm",
            norm_kwargs: Optional[Dict[str, Any]] = None,
            plain_last: bool = False,
            bias: Union[bool, List[bool]] = True,
            **kwargs):
        assert in_layers is None or in_layers == 0, "PrecomputedVar does not support in_layers."
        super(PrecomputedVar, self).__init__(
            conv=conv,
            num_hops=num_hops,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            in_layers=in_layers,
            out_layers=out_layers,
            dropout_lin=dropout_lin,
            dropout_conv=dropout_conv,
            act=act,
            act_first=act_first,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
            plain_last=plain_last,
            bias=bias,
            **kwargs
        )

    def convolute(self,
        x: Tensor,
        edge_index: Adj,
        batch: OptTensor = None,
        batch_size: Optional[int] = None,
    ) -> list:
        r"""Decoupled propagation step for calling the convolutional module.
            `self._forward()` should not contain derivable computations.
        Returns:
            embed (Tensor): List of precomputed node embeddings of each hop.
                Each shape is :math:`(|\mathcal{V}|, F, len(convs)+1)`.
        """
        conv_mat = self.get_forward_mat(x, edge_index, comp_scheme='convolute')

        xi = conv_mat['x'] if 'x' in conv_mat else conv_mat['out']
        xs = [xi]
        for conv in self.convs:
            conv.comp_scheme = 'convolute'
            conv_mat = conv._forward(**conv_mat)
            xi = conv_mat['x'] if 'x' in conv_mat else conv_mat['out']
            xs.append(xi)

        return torch.stack(xs, dim=xs[0].dim())

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
        conv_mat = self.get_forward_mat(xs[..., 0], None, comp_scheme='forward')
        for k, conv in enumerate(self.convs):
            conv.comp_scheme = 'forward'
            key = 'x' if 'x' in conv._forward.__code__.co_varnames else 'out'
            conv_mat[key] = xs[..., k+1]
            conv_mat = conv(**conv_mat)
        out = conv_mat['out']
        if self.out_layers > 0:
            out = self.out_mlp(out, batch=batch, batch_size=batch_size)
        return out
