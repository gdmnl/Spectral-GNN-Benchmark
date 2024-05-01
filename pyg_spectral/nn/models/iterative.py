import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear

from pyg_spectral.nn.models.base_nn import BaseNN
from pyg_spectral.utils import load_import


class Iterative(BaseNN):
    r"""Iterative structure with matrix transformation each hop of propagation.

    Args:
        bias (bool, optional): whether learn an additive bias in conv.
        weight_initializer (str, optional): The initializer for the weight
            matrix (:obj:`"glorot"`, :obj:`"uniform"`, :obj:`"kaiming_uniform"`
            or :obj:`None`).
        bias_initializer (str, optional): The initializer for the bias vector
            (:obj:`"zeros"` or :obj:`None`).
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
        assert self.in_layers > 0 and self.out_layers > 0, "In/out MLPs are required to ensure conv shape consistency."
        bias = kwargs.pop('bias', None)
        weight_initializer = kwargs.pop('weight_initializer', 'glorot')
        bias_initializer = kwargs.pop('bias_initializer', None)

        conv_cls = load_import(conv, lib)
        convs = nn.ModuleList()
        for k in range(num_hops):
            bias_default = (k == self.num_hops - 1)
            theta = Linear(
                self.hidden_channels, self.hidden_channels,
                bias=(bias or bias_default),
                weight_initializer=weight_initializer,
                bias_initializer=bias_initializer)
            convs.append(conv_cls(num_hops=num_hops, hop=k, theta=theta, **kwargs)) # how about the original decoupled parameters?

        return convs
