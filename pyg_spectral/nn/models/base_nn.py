from typing import Any, Callable, Dict, Final, List, Optional, Union

from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.models import MLP


class myMLP(MLP):
    def __repr__(self) -> str:
        """Fix repr of MLP"""
        return super(MLP, self).__repr__()


class BaseNN(nn.Module):
    r"""Base NN structure with MLP before and after convolution layers.

    Args:
        conv (str): Name of :class:`pyg_spectral.nn.conv` module.
        num_hops (int): Total number of conv hops.
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        in_layers (int): Number of MLP layers before conv.
        out_layers (int): Number of MLP layers after conv.
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act, act_first, act_kwargs, norm, norm_kwargs, plain_last, bias:
            args for :class:`pyg.nn.models.MLP`.
        lib_conv (str, optional): Parent module library other than
            :class:`pyg_spectral.nn.conv`.
        **kwargs (optional): Additional arguments of the
            :class:`pyg_spectral.nn.conv` module.
    """
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = False
    supports_batch: Final[bool] = False
    supports_norm_batch: Final[bool]

    def __init__(self,
            conv: str,
            num_hops: int = 0,
            in_channels: Optional[int] = None,
            hidden_channels: Optional[int] = None,
            out_channels: Optional[int] = None,
            in_layers: Optional[int] = None,
            out_layers: Optional[int] = None,
            dropout: Union[float, List[float]] = 0.,
            act: Union[str, Callable, None] = "relu",
            act_first: bool = False,
            act_kwargs: Optional[Dict[str, Any]] = None,
            norm: Union[str, Callable, None] = "batch_norm",
            norm_kwargs: Optional[Dict[str, Any]] = None,
            plain_last: bool = False,
            bias: Union[bool, List[bool]] = True,
            **kwargs):
        super(BaseNN, self).__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.in_layers = in_layers if in_layers is not None else 0
        self.out_layers = out_layers if out_layers is not None else 0
        if isinstance(dropout, list):
            self.dropout_prop = dropout[-1]
            dropout = dropout[:-1]
        else:
            self.dropout_prop = dropout
        lib = kwargs.pop('lib_conv', 'pyg_spectral.nn.conv')

        if self.in_layers > 0:
            self.in_mlp = myMLP(
                in_channels=self.in_channels,
                hidden_channels=self.hidden_channels,
                out_channels=self.hidden_channels if self.out_layers > 0 else self.out_channels,
                num_layers=self.in_layers,
                dropout=dropout,
                act=act,
                act_first=act_first,
                act_kwargs=act_kwargs,
                norm=norm,
                norm_kwargs=norm_kwargs,
                plain_last=plain_last,
                bias=bias,)

        self.num_hops = num_hops
        self.convs = self.init_conv(
            conv=conv,
            num_hops=num_hops,
            lib=lib,
            **kwargs)
        self._set_conv_func('get_forward_mat')
        self._set_conv_func('get_propagate_mat')

        if self.out_layers > 0:
            self.out_mlp = myMLP(
                in_channels=self.hidden_channels if self.in_layers > 0 else self.in_channels,
                hidden_channels=self.hidden_channels,
                out_channels=self.out_channels,
                num_layers=self.out_layers,
                dropout=dropout,
                act=act,
                act_first=act_first,
                act_kwargs=act_kwargs,
                norm=norm,
                norm_kwargs=norm_kwargs,
                plain_last=plain_last,
                bias=bias,)

    def init_conv(self, conv: str, num_hops: int, lib: str, **kwargs) -> MessagePassing:
        raise NotImplementedError

    def _set_conv_func(self, func: str):
        # if hasattr(self.conv_cls, func) and callable(getattr(self.conv_cls, func)):
        #     setattr(self, func, getattr(self.conv_cls, func))
        if hasattr(self.convs[0], func) and callable(getattr(self.convs[0], func)):
            setattr(self, func, getattr(self.convs[0], func))
        else:
            raise NotImplementedError(f"Method '{func}' not found in {self.convs[0].__name__}!")
            # setattr(self, func, lambda x: x)

    def reset_cache(self):
        for conv in self.convs:
            if hasattr(conv, 'reset_cache') and callable(getattr(conv, 'reset_cache')):
                conv.reset_cache()

    def reset_parameters(self):
        if self.in_layers > 0:
            self.in_mlp.reset_parameters()
        if self.out_layers > 0:
            self.out_mlp.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def get_wd(self, **kwargs):
        assert 'weight_decay' in kwargs, "Weight decay not found."
        res = []
        if self.in_layers > 0:
            res.append({'params': self.in_mlp.parameters(), **kwargs})
        if self.out_layers > 0:
            res.append({'params': self.out_mlp.parameters(), **kwargs})
        kwargs['weight_decay'] = 0.0
        res.append({'params': self.convs.parameters(), **kwargs})
        return res

    def preprocess(self,
        x: Tensor,
        edge_index: Adj
    ) -> Any:
        r"""Preprocessing step that not counted in forward() overhead.
        Here mainly transforming graph adjacency to actual propagation matrix.
        """
        return self.get_propagate_mat(x, edge_index)

    def convolute(self,
        x: Tensor,
        edge_index: Adj,
        batch: OptTensor = None,
        batch_size: Optional[int] = None,
    ) -> Tensor:
        r"""Decoupled propagation step for calling the convolutional module.
        """
        # NOTE: [APPNP](https://github.com/pyg-team/pytorch_geometric/blob/master/benchmark/citation/appnp.py)
        # does not have last dropout, but exists in [GPRGNN](https://github.com/jianhao2016/GPRGNN/blob/master/src/GNN_models.py)
        x = F.dropout(x, p=self.dropout_prop, training=self.training)
        # FEATURE: batch norm

        conv_mat = self.get_forward_mat(x, edge_index)
        for conv in self.convs:
            conv_mat = conv(**conv_mat)
        return conv_mat['out']

    def forward(self,
        x: Tensor,
        edge_index: Adj,
        batch: OptTensor = None,
        batch_size: Optional[int] = None,
    ) -> Tensor:
        r"""
        Args:
            x (Tensor), edge_index (Adj): from pyg.data.Data
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
        if self.in_layers > 0:
            x = self.in_mlp(x, batch=batch, batch_size=batch_size)
        x = self.convolute(x, edge_index, batch=batch, batch_size=batch_size)
        if self.out_layers > 0:
            x = self.out_mlp(x, batch=batch, batch_size=batch_size)
        return x
