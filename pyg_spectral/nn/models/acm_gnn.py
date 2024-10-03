import torch
import torch.nn as nn

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import reset

from pyg_spectral.nn.models.base_nn import BaseNN
from pyg_spectral.utils import load_import


class ACMGNN(BaseNN):
    r"""Iterative structure for ACM conv.

    :paper: Revisiting Heterophily For Graph Neural Networks
    :paper: Complete the Missing Half: Augmenting Aggregation Filtering with Diversification for Graph Convolutional Networks
    :ref: https://github.com/SitaoLuan/ACM-GNN

    Args:
        theta_scheme (str): Channel list. "FBGNN"="low-high", "ACMGNN"="low-high-id",
            ("ACMGNN+"="low-high-id-struct", not implemented).
        weight_initializer (str, optional): The initializer for the weight.
        conv, num_hops, in_channels, hidden_channels, out_channels:
            args for :class:`BaseNN`
        in_layers, out_layers, dropout_lin, dropout_conv, lib_conv:
            args for :class:`BaseNN`
        act, act_first, act_kwargs, norm, norm_kwargs, plain_last, bias:
            args for :class:`torch_geometric.nn.models.MLP`.
        **kwargs: Additional arguments of :class:`pyg_spectral.nn.conv`.
    """
    # FEATURE: separate arch
    name = 'Iterative'
    conv_name = lambda x, args: '-'.join([x, args.theta_scheme])

    def init_conv(self,
        conv: str,
        num_hops: int,
        lib: str,
        **kwargs
    ) -> MessagePassing:
        theta_scheme = kwargs.pop('theta_scheme', 'low-high-id')
        theta_scheme = [v.strip() for v in theta_scheme.split('-')]
        weight_initializer = kwargs.pop('weight_initializer', 'uniform')
        bias_initializer = kwargs.pop('bias_initializer', None)

        conv_cls = load_import(conv, lib)
        convs = nn.ModuleList()
        for k in range(num_hops):
            in_channels = self.channel_list[self.in_layers + k]
            out_channels = self.channel_list[self.in_layers + k + 1]
            kwargs['out_channels'] = out_channels
            convs.append(conv_cls(num_hops=num_hops, hop=k, **kwargs))
            convs[-1].theta = nn.ModuleDict({sch: Linear(
                in_channels, out_channels,
                bias=False,
                weight_initializer=weight_initializer,
                bias_initializer=bias_initializer) for sch in theta_scheme})
            if hasattr(convs[-1], '_init_with_theta'):
                convs[-1]._init_with_theta()

        return convs


class ACMGNNDec(BaseNN):
    r"""Decoupled structure for ACM conv.

    :paper: Revisiting Heterophily For Graph Neural Networks
    :paper: Complete the Missing Half: Augmenting Aggregation Filtering with Diversification for Graph Convolutional Networks
    :ref: https://github.com/SitaoLuan/ACM-GNN

    Args:
        theta_scheme (str): Channel list. "FBGNN"="low-high", "ACMGNN"="low-high-id",
            ("ACMGNN+"="low-high-id-struct", not implemented).
        weight_initializer (str, optional): The initializer for the weight.
        conv, num_hops, in_channels, hidden_channels, out_channels:
            args for :class:`BaseNN`
        in_layers, out_layers, dropout_lin, dropout_conv, lib_conv:
            args for :class:`BaseNN`
        act, act_first, act_kwargs, norm, norm_kwargs, plain_last, bias:
            args for :class:`torch_geometric.nn.models.MLP`.
        **kwargs: Additional arguments of :class:`pyg_spectral.nn.conv`.
    """
    # FEATURE: separate arch
    name = 'DecoupledVar'
    conv_name = lambda x, args: '-'.join([x, args.theta_scheme])

    def init_conv(self,
        conv: str,
        num_hops: int,
        lib: str,
        **kwargs
    ) -> MessagePassing:
        theta_scheme = kwargs.pop('theta_scheme', 'low-high-id')
        theta_scheme = [v.strip() for v in theta_scheme.split('-')]
        self.theta_init = {sch: torch.ones(num_hops+1) for sch in theta_scheme}

        conv_cls = load_import(conv, lib)
        convs = nn.ModuleList()
        for k in range(num_hops+1):
            kwargs['out_channels'] = self.channel_list[self.in_layers + k + 1]
            convs.append(conv_cls(num_hops=num_hops, hop=k, **kwargs))
            convs[-1].theta = nn.ParameterDict({
                sch: nn.Parameter(self.theta_init[sch][k].clone())
                for sch in theta_scheme})
            if hasattr(convs[-1], '_init_with_theta'):
                convs[-1]._init_with_theta()
        return convs

    def reset_parameters(self):
        if self.in_layers > 0:
            self.in_mlp.reset_parameters()
        if self.out_layers > 0:
            self.out_mlp.reset_parameters()
        for k, conv in enumerate(self.convs):
            reset(conv)
            for sch in conv.theta:
                conv.theta[sch].data.fill_(1.0)
