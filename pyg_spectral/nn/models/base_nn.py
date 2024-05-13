from typing import Any, Callable, Dict, Final, List, Optional, Union

import torch
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
        dropout_lin (float, optional): Dropout probability for both MLPs.
        dropout_conv (float, optional): Dropout probability before conv.
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
        super(BaseNN, self).__init__()

        self.num_hops = num_hops
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.in_layers = in_layers if in_layers is not None else 0
        self.out_layers = out_layers if out_layers is not None else 0
        self.dropout_conv = dropout_conv
        lib = kwargs.pop('lib_conv', 'pyg_spectral.nn.conv')

        # Arrange channel list
        self.channel_list = self.init_channel_list(
            conv, in_channels, hidden_channels, out_channels, **kwargs)

        if self.in_layers > 0:
            self.in_mlp = myMLP(
                channel_list=self.channel_list[:self.in_layers+1],
                num_layers=self.in_layers,
                dropout=dropout_lin,
                act=act,
                act_first=act_first,
                act_kwargs=act_kwargs,
                norm=norm,
                norm_kwargs=norm_kwargs,
                plain_last=plain_last,
                bias=bias,)

        self.convs = self.init_conv(
            conv=conv,
            num_hops=num_hops,
            lib=lib,
            **kwargs)
        self._set_conv_func('get_forward_mat')
        self._set_conv_func('get_propagate_mat')

        if self.out_layers > 0:
            self.out_mlp = myMLP(
                channel_list=self.channel_list[self.in_layers+self.num_hops:],
                num_layers=self.out_layers,
                dropout=dropout_lin,
                act=act,
                act_first=act_first,
                act_kwargs=act_kwargs,
                norm=norm,
                norm_kwargs=norm_kwargs,
                plain_last=plain_last,
                bias=bias,)

        self.reset_parameters()

    def init_channel_list(self, conv: str, in_channels: int, hidden_channels: int, out_channels: int, **kwargs) -> List[int]:
        # assert (self.in_layers+self.out_layers > 0) or (self.in_channels == self.out_channels)
        total_layers = self.in_layers + self.num_hops + self.out_layers
        channel_list = [in_channels] + [None] * (total_layers - 1) + [out_channels]
        for i in range(self.in_layers - 1):
            # 1:in_layers-1
            channel_list[i + 1] = hidden_channels
        if self.in_layers > 0:
            channel_list[self.in_layers] = hidden_channels if self.out_layers > 0 else self.out_channels
        for i in range(self.num_hops):
            # (in_layers+1):(in_layers+num_hops)
            channel_list[self.in_layers + i + 1] = channel_list[self.in_layers]
        for i in range(self.out_layers - 1):
            # (in_layers+num_hops+1):(total_layers-1)
            channel_list[self.in_layers+self.num_hops + i + 1] = hidden_channels
        return channel_list

    def init_conv(self, conv: str, num_hops: int, lib: str, **kwargs) -> MessagePassing:
        # TODO: change to nn.Sequential
        raise NotImplementedError

    def _set_conv_func(self, func: str) -> Callable:
        # if hasattr(self.conv_cls, func) and callable(getattr(self.conv_cls, func)):
        #     setattr(self, func, getattr(self.conv_cls, func))
        if hasattr(self.convs[0], func) and callable(getattr(self.convs[0], func)):
            setattr(self, func, getattr(self.convs[0], func)) # use the first one.
            return getattr(self, func)
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

    def get_optimizer(self, dct):
        res = []
        if self.in_layers > 0:
            res.append({'params': self.in_mlp.parameters(), **dct['lin']})
        if self.out_layers > 0:
            res.append({'params': self.out_mlp.parameters(), **dct['lin']})
        res.append({'params': self.convs.parameters(), **dct['conv']})
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
        x = F.dropout(x, p=self.dropout_conv, training=self.training)
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


class BaseNNCompose(BaseNN):
    r"""Base NN structure with multiple conv channels.

    Args:
        combine (str): How to combine different channels of convs. (one of
            "sum", "sum_weighted", "cat").
        --- BaseNN Args ---
        conv (str): Name of :class:`pyg_spectral.nn.conv` module.
        num_hops (int): Total number of conv hops.
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        in_layers (int): Number of MLP layers before conv.
        out_layers (int): Number of MLP layers after conv.
        dropout_lin (float, optional): Dropout probability for both MLPs.
        dropout_conv (float, optional): Dropout probability before conv.
        act, act_first, act_kwargs, norm, norm_kwargs, plain_last, bias:
            args for :class:`pyg.nn.models.MLP`.
        lib_conv (str, optional): Parent module library other than
            :class:`pyg_spectral.nn.conv`.
        **kwargs (optional): Additional arguments of the
            :class:`pyg_spectral.nn.conv` module.
    """

    def init_channel_list(self, conv: str, in_channels: int, hidden_channels: int, out_channels: int, **kwargs) -> List[int]:
        """
            self.channel_list: width for each conv channel
        """
        self.combine = kwargs.pop('combine', 'sum')

        # assert (self.in_layers+self.out_layers > 0) or (self.in_channels == self.out_channels)
        total_layers = self.in_layers + self.num_hops + self.out_layers
        channel_list = [in_channels] + [None] * (total_layers - 1) + [out_channels]
        for i in range(self.in_layers - 1):
            # 1:in_layers-1
            channel_list[i + 1] = hidden_channels
        if self.in_layers > 0:
            channel_list[self.in_layers] = hidden_channels if self.out_layers > 0 else self.out_channels

        for i in range(self.num_hops):
            # (in_layers+1):(in_layers+num_hops)
            channel_list[self.in_layers + i + 1] = channel_list[self.in_layers]
        if self.combine in ['cat']:
            channel_list[self.in_layers + self.num_hops] *= (conv.count(',') + 1)

        for i in range(self.out_layers - 1):
            # (in_layers+num_hops+1):(total_layers-1)
            channel_list[self.in_layers+self.num_hops + i + 1] = hidden_channels
        return channel_list

    def _set_conv_func(self, func: str) -> List[Callable]:
        # NOTE: return a list, not callable
        if hasattr(self.convs[0][0], func) and callable(getattr(self.convs[0][0], func)):
            lst = [getattr(channel[0], func) for channel in self.convs]
            setattr(self, func, lambda: lst)
            return getattr(self, func)
        else:
            raise NotImplementedError(f"Method '{func}' not found in {self.convs[0][0].__name__}!")
            # setattr(self, func, lambda x: x)

    def reset_cache(self):
        for channel in self.convs:
            for conv in channel:
                if hasattr(conv, 'reset_cache') and callable(getattr(conv, 'reset_cache')):
                    conv.reset_cache()

    def reset_parameters(self):
        if self.in_layers > 0:
            self.in_mlp.reset_parameters()
        if self.out_layers > 0:
            self.out_mlp.reset_parameters()
        for channel in self.convs:
            for conv in channel:
                conv.reset_parameters()
        if hasattr(self, 'gamma'):
            nn.init.ones_(self.gamma)

    def preprocess(self,
        x: Tensor,
        edge_index: Adj
    ) -> Any:
        r"""Preprocessing step that not counted in forward() overhead.
        Here mainly transforming graph adjacency to actual propagation matrix.
        """
        return [f(x, edge_index) for f in self.get_propagate_mat()]

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
        x = F.dropout(x, p=self.dropout_conv, training=self.training)
        # FEATURE: batch norm

        out = None
        conv_mats = self.get_forward_mat()
        for i, channel in enumerate(self.convs):
            conv_mat = conv_mats[i](x, edge_index)
            for conv in channel:
                conv_mat = conv(**conv_mat)

            if i == 0:
                out = conv_mat['out']
            else:
                if self.combine == 'sum':
                    out = out + conv_mat['out']
                elif self.combine == 'sum_weighted':
                    out = out + self.gamma[i] * conv_mat['out']
                elif self.combine == 'cat':
                    out = torch.cat((out, conv_mat['out']), dim=-1)
        return out
