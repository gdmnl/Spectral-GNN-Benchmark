import torch
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset

from pyg_spectral.nn.models.base_nn import BaseNN
from pyg_spectral.nn.models.decoupled import gen_theta
from pyg_spectral.utils import load_import


class AdaGNN(BaseNN):
    r"""Decoupled structure with diag transformation each hop of propagation.
    paper: AdaGNN: Graph Neural Networks with Adaptive Frequency Response Filter
    ref: https://github.com/yushundong/AdaGNN

    Args:
        --- BaseNN Args ---
        conv (str): Name of :class:`pyg_spectral.nn.conv` module.
        num_hops (int): Total number of conv hops.
        in_channels (int): Size of each input sample.
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
        self.theta_scheme = kwargs.pop('theta_scheme', 'ones')
        self.theta_param = kwargs.pop('theta_param', 1.0)

        conv_cls = load_import(conv, lib)
        convs = nn.ModuleList([
            conv_cls(num_hops=num_hops, hop=k,
                     **kwargs) for k in range(num_hops)])
        for k, convk in enumerate(convs):
            # hop k, input:output size: [in_layers+k:in_layers+k+1]
            convk.register_parameter('theta', nn.Parameter(torch.FloatTensor(self.channel_list[self.in_layers+k+1])))
            if hasattr(convk, '_init_with_theta'):
                convk._init_with_theta()
        return convs

    def reset_parameters(self):
        if self.in_layers > 0:
            self.in_mlp.reset_parameters()
        if self.out_layers > 0:
            self.out_mlp.reset_parameters()

        for k, conv in enumerate(self.convs):
            f = conv.theta.size(0)
            if self.theta_scheme.startswith(('uniform', 'normal')):
                thetas = gen_theta(f-1, self.theta_scheme, self.theta_param)
            else:
                thetas = gen_theta(self.num_hops, self.theta_scheme, self.theta_param)[k].repeat(f)
            conv.theta.data = thetas
            reset(conv)
