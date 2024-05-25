import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing

from pyg_spectral.nn.models.base_nn import BaseNNCompose
from pyg_spectral.nn.models.decoupled import gen_theta
from pyg_spectral.utils import load_import


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
        if self.combine == 'sum_weighted':
            self.gamma = nn.Parameter(torch.ones(len(conv), 1))

        for c, scheme, param in zip(conv, theta_schemes, theta_params):
            theta = gen_theta(num_hops, scheme, param)
            conv_cls = load_import(c, lib)
            kwargs_c = {}
            for k, v in kwargs.items():
                # Find required arguments for current conv class
                if k in conv_cls.__init__.__code__.co_varnames:
                    if isinstance(v, list) and len(v) == len(conv):
                        kwargs_c[k] = v[c]
                    else:
                        kwargs_c[k] = v

            # NOTE: k=0 layer explicitly handles x without propagation. So there is
            # (num_hops+1) conv layers in total.
            convs.append(nn.ModuleList([
                conv_cls(num_hops=num_hops, hop=k, **kwargs_c) for k in range(num_hops+1)]))
            for k, convk in enumerate(convs[-1]):
                convk.register_buffer('theta', theta[k].clone())
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
        if self.combine == 'sum_weighted':
            self.gamma = nn.Parameter(torch.ones(len(conv), 1))

        self.theta_init = []
        for c, scheme, param in zip(conv, theta_schemes, theta_params):
            self.theta_init.append(gen_theta(num_hops, scheme, param))
            conv_cls = load_import(c, lib)
            kwargs_c = {}
            for k, v in kwargs.items():
                # Find required arguments for current conv class
                if k in conv_cls.__init__.__code__.co_varnames:
                    if isinstance(v, list) and len(v) == len(conv):
                        kwargs_c[k] = v[c]
                    else:
                        kwargs_c[k] = v

            # NOTE: k=0 layer explicitly handles x without propagation. So there is
            # (num_hops+1) conv layers in total.
            convs.append(nn.ModuleList([
                conv_cls(num_hops=num_hops, hop=k, theta=nn.Parameter(self.theta_init[-1][k]), **kwargs_c) for k in range(num_hops+1)]))
            for k, convk in enumerate(convs[-1]):
                convk.register_parameter('theta', nn.Parameter(self.theta_init[-1][k].clone()))
        return convs

    def reset_parameters(self):
        if self.in_layers > 0:
            self.in_mlp.reset_parameters()
        if self.out_layers > 0:
            self.out_mlp.reset_parameters()
        for i, channel in enumerate(self.convs):
            for k, conv in enumerate(channel):
                conv.reset_parameters()
                conv.theta.data = self.theta_init[i][k].clone()
        if hasattr(self, 'gamma'):
            nn.init.ones_(self.gamma)
