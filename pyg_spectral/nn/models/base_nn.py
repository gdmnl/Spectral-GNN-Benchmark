from typing import Any, Callable, Final, NewType
ParamTuple = NewType('ParamTuple', tuple[str, tuple, dict[str, Any], Callable[[Any], str]])

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.models import MLP
from torch_geometric.nn.inits import reset


MODEL_REGI_INIT = {k: {} for k in ['name', 'conv_name', 'module', 'pargs', 'pargs_default', 'param']}


class myMLP(MLP):
    def __repr__(self) -> str:
        """Fix repr of MLP"""
        return super(MLP, self).__repr__()


class BaseNN(nn.Module):
    r"""Base NN structure with MLP before and after convolution layers.

    Args:
        conv: Name of :class:`pyg_spectral.nn.conv` module.
        num_hops: Total number of conv hops.
        in_channels: Size of each input sample.
        hidden_channels: Size of each hidden sample.
        out_channels: Size of each output sample.
        in_layers: Number of MLP layers before conv.
        out_layers: Number of MLP layers after conv.
        dropout_lin: Dropout probability for both MLPs.
        dropout_conv: Dropout probability before conv.
        act, act_first, act_kwargs, norm, norm_kwargs, plain_last, bias:
            args for :class:`torch_geometric.nn.models.MLP`.
        lib_conv: Parent module library other than :class:`pyg_spectral.nn.conv`.
        **kwargs: Additional arguments of :class:`pyg_spectral.nn.conv`.
    """
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = False
    supports_norm_batch: Final[bool]
    supports_batch: bool
    name: str
    conv_name: Callable[[str, Any], str] = lambda x, args: x
    pargs: list[str] = ['conv', 'num_hops', 'in_layers', 'out_layers',
                        'in_channels', 'hidden_channels', 'out_channels',
                        'dropout_lin', 'dropout_conv',]
    param: dict[str, ParamTuple] = {
            'num_hops':     ('int', (2, 30), {'step': 2}, lambda x: x),
            'in_layers':    ('int', (1, 3), {}, lambda x: x),
            'out_layers':   ('int', (1, 3), {}, lambda x: x),
            'hidden_channels':  ('categorical', ([16, 32, 64, 128, 256],), {}, lambda x: x),
            'dropout_lin':  ('float', (0.0, 0.9), {'step': 0.1}, lambda x: round(x, 2)),
            'dropout_conv': ('float', (0.0, 0.9), {'step': 0.1}, lambda x: round(x, 2)),
    }

    @classmethod
    def register_classes(cls, registry: dict[str, dict[str, Any]] = None) -> dict:
        r"""Register args for all subclass.

        Args:
            name (dict[str, str]): Model class logging path name.
            conv_name (dict[str, Callable[[str, Any], str]]): Wrap conv logging path name.
            module (dict[str, str]): Module for importing the model.
            pargs (dict[str, list[str]]): Model arguments from argparse.
            pargs_default (dict[str, dict[str, Any]]): Default values for model arguments. Not recommended.
            param (dict[str, dict[str, ParamTuple]]): Model parameters to tune.

                * (str) parameter type,
                * (tuple) args for :func:`optuna.trial.suggest_<type>`,
                * (dict) kwargs for :func:`optuna.trial.suggest_<type>`,
                * (callable) format function to str.
        """
        if registry is None:
            registry = MODEL_REGI_INIT

        for subcls in cls.__subclasses__():
            subname = subcls.__name__
            # Traverse the MRO and accumulate args from parent classes
            registry['pargs'][subname], registry['param'][subname] = [], {}
            for basecls in subcls.mro():
                if hasattr(basecls, 'pargs'):
                    registry['pargs'][subname].extend(basecls.pargs)
                if hasattr(basecls, 'pargs_default'):
                    registry['pargs_default'][subname].update(basecls.pargs_default)
                if hasattr(basecls, 'param'):
                    registry['param'][subname].update(basecls.param)
            if hasattr(subcls, 'name'):
                registry['name'][subname] = subcls.name
            if hasattr(subcls, 'conv_name'):
                registry['conv_name'][subname] = subcls.conv_name
            registry['module'][subname] = '.'.join(cls.__module__.split('.')[:-1])

            subcls.register_classes(registry)
        return registry

    def __init__(self,
            conv: str,
            num_hops: int = 0,
            in_channels: int | None = None,
            hidden_channels: int | None = None,
            out_channels: int | None = None,
            in_layers: int | None = None,
            out_layers: int | None = None,
            dropout_lin: float | list[float] = 0.,
            dropout_conv: float = 0.,
            act: str | Callable | None = "relu",
            act_first: bool = False,
            act_kwargs: dict[str, Any | None] = None,
            norm: str | Callable | None = "batch_norm",
            norm_kwargs: dict[str, Any | None] = None,
            plain_last: bool = False,
            bias: bool | list[bool] = True,
            **kwargs):
        super(BaseNN, self).__init__()

        self.num_hops = num_hops
        self.conv_layers = num_hops + 1
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
        self._set_conv_attr('get_forward_mat')
        self._set_conv_attr('get_propagate_mat')
        self._set_conv_attr('propagate_be')

        if self.out_layers > 0:
            self.out_mlp = myMLP(
                channel_list=self.channel_list[self.in_layers+self.conv_layers:],
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

    def init_channel_list(self, conv: str, in_channels: int, hidden_channels: int, out_channels: int, **kwargs) -> list[int]:
        # assert (self.in_layers+self.out_layers > 0) or (self.in_channels == self.out_channels)
        total_layers = self.in_layers + self.conv_layers + self.out_layers
        channel_list = [in_channels] + [None] * (total_layers - 1) + [out_channels]
        for i in range(self.in_layers - 1):
            # 1:in_layers-1
            channel_list[i + 1] = hidden_channels
        if self.in_layers > 0:
            channel_list[self.in_layers] = hidden_channels if self.out_layers > 0 else self.out_channels
        for i in range(self.conv_layers):
            # (in_layers+1):(in_layers+conv_layers)
            channel_list[self.in_layers + i + 1] = channel_list[self.in_layers]
        for i in range(self.out_layers - 1):
            # (in_layers+conv_layers+1):(total_layers-1)
            channel_list[self.in_layers+self.conv_layers + i + 1] = hidden_channels
        return channel_list

    def init_conv(self, conv: str, num_hops: int, lib: str, **kwargs) -> MessagePassing:
        raise NotImplementedError

    def _set_conv_attr(self, key: str) -> Callable:
        # if hasattr(self.conv_cls, key):
        #     setattr(self, key, getattr(self.conv_cls, key))
        if hasattr(self.convs[0], key):
            setattr(self, key, getattr(self.convs[0], key)) # use the first layer
            return getattr(self, key)
        else:
            raise NotImplementedError(f"Attribute '{key}' not found in {self.convs[0].__class__}!")
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
            reset(conv)

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
        r"""Preprocessing step that not counted in :meth:`forward()` overhead.
        Here mainly transforming graph adjacency to actual propagation matrix.
        """
        return self.get_propagate_mat(x, edge_index)

    def convolute(self,
        x: Tensor,
        edge_index: Adj,
        batch: OptTensor = None,
        batch_size: int | None = None,
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
        batch_size: int | None = None,
    ) -> Tensor:
        r"""
        Args:
            x, edge_index: from :class:`torch_geometric.data.Data`
            batch: The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example.
                Only needs to be passed in case the underlying normalization
                layers require the :obj:`batch` information.
            batch_size: The number of examples :math:`B`.
                Automatically calculated if not given.
                Only needs to be passed in case the underlying normalization
                layers require the :obj:`batch` information.
        """
        if self.in_layers > 0:
            x = self.in_mlp(x, batch=batch, batch_size=batch_size)
        x = self.convolute(x, edge_index, batch=batch, batch_size=batch_size)
        if self.out_layers > 0:
            x = self.out_mlp(x, batch=batch, batch_size=batch_size)
        return x


class BaseLPNN(BaseNN):
    def __init__(self,
            conv: str,
            num_hops: int = 0,
            in_channels: int | None = None,
            hidden_channels: int | None = None,
            out_channels: int | None = None,
            in_layers: int | None = None,
            out_layers: int | None = None,
            dropout_lin: float | list[float] = 0.,
            dropout_conv: float = 0.,
            act: str | Callable | None = "relu",
            act_first: bool = False,
            act_kwargs: dict[str, Any | None] = None,
            norm: str | Callable | None = "batch_norm",
            norm_kwargs: dict[str, Any | None] = None,
            plain_last: bool = False,
            bias: bool | list[bool] = True,
            **kwargs):
        super(BaseLPNN, self).__init__(
            conv=conv,
            num_hops=num_hops,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            in_layers=in_layers,
            out_layers=out_layers,
            dropout_lin=dropout_lin,
            dropout_conv=dropout_conv,
            act=act,
            act_first=act_first,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
            plain_last=True,
            bias=bias,
            **kwargs)
        self.out_channels = out_channels
        self.desc_layers = 1

        if self.desc_layers > 0:
            self.desc_mlp = myMLP(
                channel_list=[hidden_channels] * self.desc_layers + [out_channels],
                num_layers=self.desc_layers,
                dropout=0.0,
                act=act,
                act_first=act_first,
                act_kwargs=act_kwargs,
                norm=None,
                norm_kwargs=norm_kwargs,
                plain_last=plain_last,
                bias=bias,)
            self.desc_mlp.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'desc_layers') and self.desc_layers > 0:
            self.desc_mlp.reset_parameters()
        super(BaseLPNN, self).reset_parameters()

    def get_optimizer(self, dct):
        res = []
        if self.in_layers > 0:
            res.append({'params': self.in_mlp.parameters(), **dct['lin']})
        if self.out_layers > 0:
            res.append({'params': self.out_mlp.parameters(), **dct['lin']})
        if self.desc_layers > 0:
            res.append({'params': self.desc_mlp.parameters(), **dct['lin']})
        res.append({'params': self.convs.parameters(), **dct['conv']})
        return res

    def decode(self,
        z: Tensor,
        edge_label_index: Adj,
        batch: OptTensor = None,
        batch_size: int | None = None,
    ) -> Tensor:
        x = z[edge_label_index[0]] * z[edge_label_index[1]]
        x = self.desc_mlp(x, batch=batch, batch_size=batch_size)
        return x.flatten()


class BaseNNCompose(BaseNN):
    r"""Base NN structure with multiple conv channels.

    Args:
        combine (str): How to combine different channels of convs. (:obj:`sum`,
            :obj:`sum_weighted`, or :obj:`cat`).
        conv: Name of :class:`pyg_spectral.nn.conv` module.
        num_hops: Total number of conv hops.
        in_channels: Size of each input sample.
        hidden_channels: Size of each hidden sample.
        out_channels: Size of each output sample.
        in_layers: Number of MLP layers before conv.
        out_layers: Number of MLP layers after conv.
        dropout_lin: Dropout probability for both MLPs.
        dropout_conv: Dropout probability before conv.
        act, act_first, act_kwargs, norm, norm_kwargs, plain_last, bias:
            args for :class:`torch_geometric.nn.models.MLP`.
        lib_conv: Parent module library other than :class:`pyg_spectral.nn.conv`.
        **kwargs: Additional arguments of :class:`pyg_spectral.nn.conv`.
    """
    pargs = ['combine']
    param = {'combine': ('categorical', ['sum', 'sum_weighted', 'cat'], {}, lambda x: x)}

    def init_channel_list(self, conv: str, in_channels: int, hidden_channels: int, out_channels: int, **kwargs) -> list[int]:
        """
        Attributes:
            channel_list: width for each conv channel
        """
        self.combine = kwargs.pop('combine', 'sum')
        n_conv = len(conv.split(','))

        # assert (self.in_layers+self.out_layers > 0) or (self.in_channels == self.out_channels)
        total_layers = self.in_layers + self.conv_layers + self.out_layers
        channel_list = [in_channels] + [None] * (total_layers - 1) + [out_channels]
        for i in range(self.in_layers - 1):
            # 1:in_layers-1
            channel_list[i + 1] = hidden_channels
        if self.in_layers > 0:
            channel_list[self.in_layers] = hidden_channels if self.out_layers > 0 else self.out_channels

        for i in range(self.conv_layers):
            # (in_layers+1):(in_layers+conv_layers)
            channel_list[self.in_layers + i + 1] = channel_list[self.in_layers]
        if self.combine in ['cat']:
            channel_list[self.in_layers + self.conv_layers] *= n_conv

        for i in range(self.out_layers - 1):
            # (in_layers+conv_layers+1):(total_layers-1)
            channel_list[self.in_layers+self.conv_layers + i + 1] = hidden_channels

        if self.combine == 'sum_weighted':
            self.gamma = nn.Parameter(torch.ones(n_conv, 1))
        elif self.combine == 'sum_vec':
            self.gamma = nn.Parameter(torch.ones(n_conv, channel_list[self.in_layers + self.conv_layers]))
        return channel_list

    def _set_conv_attr(self, key: str) -> list[Callable]:
        # NOTE: return a list, not callable
        if hasattr(self.convs[0][0], key):
            lst = [getattr(channel[0], key) for channel in self.convs]
            setattr(self, key, lambda: lst)
            return getattr(self, key)
        else:
            raise NotImplementedError(f"Attribute '{key}' not found in {self.convs[0][0].__name__}!")

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
                reset(conv)
        if hasattr(self, 'gamma'):
            nn.init.ones_(self.gamma)

    def get_optimizer(self, dct):
        res = []
        if self.in_layers > 0:
            res.append({'params': self.in_mlp.parameters(), **dct['lin']})
        if self.out_layers > 0:
            res.append({'params': self.out_mlp.parameters(), **dct['lin']})
        if hasattr(self, 'gamma'):
            res.append({'params': self.gamma, **dct['conv']})
        res.append({'params': self.convs.parameters(), **dct['conv']})
        return res

    def preprocess(self,
        x: Tensor,
        edge_index: Adj
    ) -> Any:
        r"""Preprocessing step that not counted in :meth:`forward()` overhead.
        Here mainly transforming graph adjacency to actual propagation matrix.
        """
        return [f(x, edge_index) for f in self.get_propagate_mat()]

    def convolute(self,
        x: Tensor,
        edge_index: Adj,
        batch: OptTensor = None,
        batch_size: int | None = None,
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
                elif self.combine in ['sum_weighted', 'sum_vec']:
                    out = out + self.gamma[i] * conv_mat['out']
                elif self.combine == 'cat':
                    out = torch.cat((out, conv_mat['out']), dim=-1)
        return out
