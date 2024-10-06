import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear

from pyg_spectral.nn.models.base_nn import BaseNN, BaseNNCompose
from pyg_spectral.utils import load_import


class Iterative(BaseNN):
    r"""Iterative structure with matrix transformation each hop of propagation.

    Args:
        bias (bool | None): whether learn an additive bias in conv.
        weight_initializer (str | None): The initializer for the weight
            matrix (:obj:`"glorot"`, :obj:`"uniform"`, :obj:`"kaiming_uniform"`,
            or :obj:`None`).
        bias_initializer (str | None): The initializer for the bias vector
            (:obj:`"zeros"` or :obj:`None`).
        conv, num_hops, in_channels, hidden_channels, out_channels:
            args for :class:`BaseNN`
        in_layers, out_layers, dropout_lin, dropout_conv, lib_conv:
            args for :class:`BaseNN`
        act, act_first, act_kwargs, norm, norm_kwargs, plain_last, bias:
            args for :class:`torch_geometric.nn.models.MLP`.
        **kwargs: Additional arguments of :class:`pyg_spectral.nn.conv`.
    """
    name = 'Iterative'

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
        bias (bool | None): whether learn an additive bias in conv.
        weight_initializer (str | None): The initializer for the weight
            matrix (:obj:`"glorot"`, :obj:`"uniform"`, :obj:`"kaiming_uniform"`,
            or :obj:`None`).
        bias_initializer (str | None): The initializer for the bias vector
            (:obj:`"zeros"` or :obj:`None`).
        combine: How to combine different channels of convs. (:obj:`sum`,
            :obj:`sum_weighted`, or :obj:`cat`).
        conv, num_hops, in_channels, hidden_channels, out_channels:
            args for :class:`BaseNN`
        in_layers, out_layers, dropout_lin, dropout_conv, lib_conv:
            args for :class:`BaseNN`
        act, act_first, act_kwargs, norm, norm_kwargs, plain_last, bias:
            args for :class:`torch_geometric.nn.models.MLP`.
        **kwargs: Additional arguments of :class:`pyg_spectral.nn.conv`.
    """
    name = 'Iterative'

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


class IterativeFixed(Iterative):
    name = 'IterativeFixed'
    conv_name = lambda x, args: '-'.join([x, args.theta_scheme])


class IterativeFixedCompose(IterativeCompose):
    name = 'IterativeFixed'
    conv_name = lambda x, args: '-'.join([x, args.theta_scheme])
