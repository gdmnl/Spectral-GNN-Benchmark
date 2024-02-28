from typing import Any, Callable, Dict, Final, List, Optional, Union
from torch_geometric.typing import Adj, OptTensor

import torch
import torch.nn as nn
from torch_geometric.nn.models import MLP

from pyg_spectral.utils import load_import


class myMLP(MLP):
    def __repr__(self) -> str:
        return super(MLP, self).__repr__()


class DecPostMLP(nn.Module):
    r"""Post-propagation decoupled model.

    Args:
        conv (str): Name of :class:`pyg_spectral.nn.conv` module.
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        num_layers (int): Number of message passing layers.
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        plain_last (bool, optional): If set to :obj:`False`, will apply
            non-linearity, batch normalization and dropout to the last layer as
            well. (default: :obj:`True`)
        bias (bool or List[bool], optional): If set to :obj:`False`, the module
            will not learn additive biases. If a list is provided, sets the
            bias per layer. (default: :obj:`True`)
        lib_conv (str, optional): Parent module library other than
            :class:`pyg_spectral.nn.conv`.
        **kwargs (optional): Additional arguments of the
            :class:`torch.nn.Module`.
    """
    supports_edge_weight: Final[bool] = True
    supports_edge_attr: Final[bool] = False
    supports_norm_batch: Final[bool]

    def __init__(self,
            conv: str,
            in_channels: Optional[int] = None,
            hidden_channels: Optional[int] = None,
            out_channels: Optional[int] = None,
            num_layers: Optional[int] = None,
            dropout: Union[float, List[float]] = 0.,
            act: Union[str, Callable, None] = "relu",
            act_first: bool = False,
            act_kwargs: Optional[Dict[str, Any]] = None,
            norm: Union[str, Callable, None] = "batch_norm",
            norm_kwargs: Optional[Dict[str, Any]] = None,
            plain_last: bool = True,
            bias: Union[bool, List[bool]] = True,
            **kwargs):
        super(DecPostMLP, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        # TODO: check whether to apply act and norn at last
        self.plain_last = False
        lib = kwargs.pop('lib_conv', 'pyg_spectral.nn.conv')

        self.mlp = myMLP(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            act=act,
            act_first=act_first,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
            plain_last=self.plain_last,
            bias=bias,)
        self.conv = load_import(conv, lib)(**kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        self.mlp.reset_parameters()
        self.conv.reset_parameters()

    def forward(self,
        x: torch.Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
        batch_size: Optional[int] = None,
        num_sampled_nodes_per_hop: Optional[List[int]] = None,
        num_sampled_edges_per_hop: Optional[List[int]] = None,
    ) -> torch.Tensor:
        x = self.mlp(x)
        # TODO: if not plain_last, apply dropout

        # TODO: if it possible to decouple prop (say on CPU)
        if self.supports_edge_weight and self.supports_edge_attr:
            x = self.conv(x, edge_index, edge_weight=edge_weight,
                        edge_attr=edge_attr)
        elif self.supports_edge_weight:
            x = self.conv(x, edge_index, edge_weight=edge_weight)
        elif self.supports_edge_attr:
            x = self.conv(x, edge_index, edge_attr=edge_attr)
        else:
            x = self.conv(x, edge_index)

        return x
