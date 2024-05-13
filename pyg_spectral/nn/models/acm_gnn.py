import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear

from pyg_spectral.nn.models.base_nn import BaseNN
from pyg_spectral.utils import load_import


class ACMGNN(BaseNN):
    r"""Iterative structure for ACM conv.
    paper: Revisiting Heterophily For Graph Neural Networks
    paper: Complete the Missing Half: Augmenting Aggregation Filtering with Diversification for Graph Convolutional Networks
    ref: https://github.com/SitaoLuan/ACM-GNN

    Args:
        theta_scheme (str): Channel list. "FBGNN"="low-high", "ACMGNN"="low-high-id",
            ("ACMGNN+"="low-high-id-struct", not implemented).
        weight_initializer (str, optional): The initializer for the weight.
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
        theta_scheme = kwargs.pop('theta_scheme', 'low-high-id')
        theta_scheme = [v.strip() for v in theta_scheme.split('-')]
        weight_initializer = kwargs.pop('weight_initializer', 'uniform')
        bias_initializer = kwargs.pop('bias_initializer', None)

        conv_cls = load_import(conv, lib)
        convs = nn.ModuleList()
        for k in range(num_hops):
            in_channels = self.channel_list[self.in_layers + k]
            out_channels = self.channel_list[self.in_layers + k + 1]
            theta = nn.ModuleDict({sch: Linear(
                in_channels, out_channels,
                bias=False,
                weight_initializer=weight_initializer,
                bias_initializer=bias_initializer) for sch in theta_scheme})
            convs.append(conv_cls(num_hops=num_hops, hop=k, theta=theta, **kwargs))

        return convs
