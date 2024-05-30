from typing import Any, Callable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset

from pyg_spectral.nn.models.base_nn import BaseNN
from pyg_spectral.nn.models.decoupled import gen_theta
from pyg_spectral.utils import load_import


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

        if self.combine == 'sum_weighted':
            self.gamma = nn.Parameter(torch.ones(len(conv), 1))
        elif self.combine == 'sum_vec':
            self.gamma = nn.Parameter(torch.ones(len(conv), channel_list[self.in_layers + self.num_hops]))
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
                elif self.combine in ['sum_weighted', 'sum_vec']:
                    out = out + self.gamma[i] * conv_mat['out']
                elif self.combine == 'cat':
                    out = torch.cat((out, conv_mat['out']), dim=-1)
        return out


class DecoupledFixedCompose(BaseNNCompose):
    r"""Decoupled structure without matrix transformation during propagation.
        Fixed scalar propagation parameters.

    Args:
        theta_scheme (List[str]): Method to generate decoupled parameters.
        theta_param (List[float], optional): Hyperparameter for the scheme.
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

    def init_conv(self,
        conv: str,
        num_hops: int,
        lib: str,
        **kwargs
    ) -> MessagePassing:
        theta_schemes = kwargs.pop('theta_scheme', 'appr')
        theta_schemes = theta_schemes.split(',')
        theta_params = kwargs.pop('theta_param', [0.5])
        conv = conv.split(',')
        convs = nn.ModuleList()

        for i, channel in enumerate(conv):
            theta = gen_theta(num_hops, theta_schemes[i], theta_params[i])
            conv_cls = load_import(channel, lib)
            kwargs_c = {}
            for k, v in kwargs.items():
                # Find required arguments for current conv class
                if k in conv_cls.__init__.__code__.co_varnames:
                    if isinstance(v, list) and len(v) == len(conv):
                        kwargs_c[k] = v[i]
                    else:
                        kwargs_c[k] = v

            # NOTE: k=0 layer explicitly handles x without propagation. So there
            # are (num_hops+1) conv layers in total.
            convs.append(nn.ModuleList([
                conv_cls(num_hops=num_hops, hop=k, **kwargs_c) for k in range(num_hops+1)]))
            for k, convk in enumerate(convs[-1]):
                convk.register_buffer('theta', theta[k].clone())
                if hasattr(convk, '_init_with_theta'):
                    convk._init_with_theta()
        return convs


class DecoupledVarCompose(BaseNNCompose):
    r"""Decoupled structure without matrix transformation during propagation.
        Learnable scalar propagation parameters.

    Args:
        theta_scheme (List[str]): Method to generate decoupled parameters.
        theta_param (List[float], optional): Hyperparameter for the scheme.
        combine (str): How to combine different channels of convs. (one of
            "sum", "sum_weighted", "cat").
        --- BaseNN Args ---
        conv (List[str]): Name of :class:`pyg_spectral.nn.conv` module.
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

    def init_conv(self,
        conv: str,
        num_hops: int,
        lib: str,
        **kwargs
    ) -> MessagePassing:
        conv = conv.split(',')
        theta_schemes = kwargs.pop('theta_scheme', ['ones'])
        theta_schemes = theta_schemes.split(',')
        if len(theta_schemes) == 1:
            theta_schemes = theta_schemes * len(conv)
        theta_params = kwargs.pop('theta_param', [1.0])
        if not isinstance(theta_params, list):
            theta_params = [theta_params]
        if len(theta_params) == 1:
            theta_params = theta_params * len(conv)
        convs = nn.ModuleList()

        self.theta_init = []

        for i, channel in enumerate(conv):
            self.theta_init.append(gen_theta(num_hops, theta_schemes[i], theta_params[i]))
            conv_cls = load_import(channel, lib)
            kwargs_c = {}
            for k, v in kwargs.items():
                # Find required arguments for current conv class
                if k in conv_cls.__init__.__code__.co_varnames:
                    if isinstance(v, list) and len(v) == len(conv):
                        kwargs_c[k] = v[i]
                    else:
                        kwargs_c[k] = v

            # NOTE: k=0 layer explicitly handles x without propagation. So there
            # are (num_hops+1) conv layers in total.
            convs.append(nn.ModuleList([
                conv_cls(num_hops=num_hops, hop=k, **kwargs_c) for k in range(num_hops+1)]))
            for k, convk in enumerate(convs[-1]):
                convk.register_parameter('theta', nn.Parameter(self.theta_init[-1][k].clone()))
                if hasattr(convk, '_init_with_theta'):
                    convk._init_with_theta()
        return convs

    def reset_parameters(self):
        if self.in_layers > 0:
            self.in_mlp.reset_parameters()
        if self.out_layers > 0:
            self.out_mlp.reset_parameters()
        for i, channel in enumerate(self.convs):
            for k, conv in enumerate(channel):
                reset(conv)
                conv.theta.data = self.theta_init[i][k].clone()
        if hasattr(self, 'gamma'):
            nn.init.ones_(self.gamma)
