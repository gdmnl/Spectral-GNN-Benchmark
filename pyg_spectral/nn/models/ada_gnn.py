import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing

from pyg_spectral.nn.models.base_nn import BaseNN
from pyg_spectral.utils import load_import


class AdaGNN(BaseNN):
    r"""Decoupled structure with diag transformation each hop of propagation.
    paper: AdaGNN: Graph Neural Networks with Adaptive Frequency Response Filter
    ref: https://github.com/yushundong/AdaGNN

    Args:
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
        conv_cls = load_import(conv, lib)
        return nn.ModuleList([
            conv_cls(num_hops=num_hops, hop=k,
                     theta=nn.Parameter(torch.FloatTensor(self.channel_list[self.in_layers+k])),
                     **kwargs) for k in range(num_hops)])

    def reset_parameters(self):
        if self.in_layers > 0:
            self.in_mlp.reset_parameters()
        if self.out_layers > 0:
            self.out_mlp.reset_parameters()
        for k, conv in enumerate(self.convs):
            conv.reset_parameters()
            nn.init.normal_(conv.theta, mean=0, std=1e-7)
