import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear

from pyg_spectral.nn.models.base_nn import BaseNN, BaseNNCompose
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
        bias = kwargs.pop('bias', None)
        weight_initializer = kwargs.pop('weight_initializer', 'glorot')
        bias_initializer = kwargs.pop('bias_initializer', None)

        conv_cls = load_import(conv, lib)
        convs = nn.ModuleList()
        # NOTE: There are (num_hops+1) conv layers in total.
        for k in range(num_hops+1):
            bias_default = (k == num_hops)
            convs.append(conv_cls(num_hops=num_hops, hop=k, **kwargs))
            # hop k, input:output size: [in_layers+k:in_layers+k+1]
            convs[-1].theta = Linear(
                self.channel_list[self.in_layers+k],
                self.channel_list[self.in_layers+k+1],
                bias=(bias or bias_default),
                weight_initializer=weight_initializer,
                bias_initializer=bias_initializer)
            if hasattr(convs[-1], '_init_with_theta'):
                convs[-1]._init_with_theta()

        return convs


class IterativeCompose(BaseNNCompose):
    r"""Iterative structure with matrix transformation each hop of propagation.

    Args:
        bias (bool, optional): whether learn an additive bias in conv.
        weight_initializer (str, optional): The initializer for the weight
            matrix (:obj:`"glorot"`, :obj:`"uniform"`, :obj:`"kaiming_uniform"`
            or :obj:`None`).
        bias_initializer (str, optional): The initializer for the bias vector
            (:obj:`"zeros"` or :obj:`None`).
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
        convs = nn.ModuleList()

        bias = kwargs.pop('bias', None)
        weight_initializer = kwargs.pop('weight_initializer', 'glorot')
        bias_initializer = kwargs.pop('bias_initializer', None)

        for i, channel in enumerate(conv):
            conv_cls = load_import(channel, lib)
            kwargs_c = {}
            for k, v in kwargs.items():
                # Find required arguments for current conv class
                if k in conv_cls.__init__.__code__.co_varnames:
                    if isinstance(v, list) and len(v) == len(conv):
                        kwargs_c[k] = v[i]
                    else:
                        kwargs_c[k] = v

            convsi = nn.ModuleList()
            # NOTE: There are (num_hops+1) conv layers in total.
            for k in range(num_hops+1):
                bias_default = (k == num_hops)
                convsi.append(conv_cls(num_hops=num_hops, hop=k, **kwargs_c))

                out_channels = self.channel_list[self.in_layers+k+1]
                if k == num_hops and self.combine in ['cat']:
                    out_channels = out_channels // len(conv)
                convsi[-1].theta = Linear(
                    self.channel_list[self.in_layers+k],
                    out_channels,
                    bias=(bias or bias_default),
                    weight_initializer=weight_initializer,
                    bias_initializer=bias_initializer)
                if hasattr(convsi[-1], '_init_with_theta'):
                    convsi[-1]._init_with_theta()
            convs.append(convsi)
        return convs
