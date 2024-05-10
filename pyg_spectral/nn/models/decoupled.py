from typing import List, Union

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn

from torch_geometric.nn.conv import MessagePassing

from pyg_spectral.nn.models.base_nn import BaseNN
from pyg_spectral.utils import load_import


def gen_theta(num_hops: int, scheme: str, param: Union[float, List[float]] = None) -> Tensor:
    r"""Generate list of hop parameters based on given scheme.

    Args:
        num_hops (int): Total number of hops.
        scheme (str): Method to generate parameters.
            - `ones`: all-same, :math:`\theta_k = p`.
            - `impulse`: K-hop, :math:`\theta_K = p, else 0`.
            - `mono`: Monomial, :math:`\theta_k = (1-p)/K, \theta_0 = p`.
            - 'appr': Approximate PPR, :math:`\theta_k = p (1 - p)^k`.
            - 'nappr': Negative PPR, :math:`\theta_k = p^k`.
            - 'hk': Heat Kernel, :math:`\theta_k = e^{-p}p^k / k!`.
            - 'gaussian': Graph Gaussian Kernel, :math:`theta_k = p^{k} / k!`.
            - 'uniform': Random uniform distribution.
            - 'normal': Random Gaussian distribution.
            - 'custom': Custom list of hop parameters.
        param (float, optional): Hyperparameter for the scheme.
            - `ones`: Value.
            - 'impulse': Value.
            - 'mono': Decay factor, :math:`p \in [0, 1]`.
            - 'appr': Decay factor, :math:`p \in [0, 1]`.
            - 'nappr': Decay factor, :math:`p \in [-1, 1]`.
            - 'hk': Decay factor, :math:`p > 0`.
            - 'gaussian': Decay factor, :math:`p > 0`.
            - 'uniform': Distribution bound.
            - 'normal': Distribution variance.
            - 'custom': Float list of hop parameters.

    Returns:
        theta (Tensor): Lenth (num_hops+1) list of hop parameters.
    """
    assert num_hops > 0, 'num_hops should be a positive integer'
    if scheme == 'ones':
        param = param if param is not None else 1.0
        return torch.ones(num_hops+1, dtype=torch.float) * param
    elif scheme == 'impulse':
        param = param if param is not None else 1.0
        theta = torch.zeros(num_hops+1, dtype=torch.float)
        theta[num_hops] = param
        return theta
    elif scheme == 'mono':
        param = param if param is not None else 0.5
        theta = torch.zeros(num_hops+1, dtype=torch.float)
        theta[0] = param
        theta[1:] = (1 - param) / num_hops
        return theta
    elif scheme == 'appr':
        param = param if param is not None else 0.5
        # theta[-1] = (1 - param) ** num_hops
        return param * (1 - param) ** torch.arange(num_hops+1)
    elif scheme == 'nappr':
        param = param if param is not None else 0.5
        theta = param ** torch.arange(num_hops+1)
        return theta/torch.norm(theta, p=1)
    elif scheme == 'hk':
        param = param if param is not None else 1.0
        k = torch.arange(num_hops+1)
        factorial = torch.tensor([np.math.factorial(i) for i in range(num_hops+1)])
        return torch.exp(-param) * (param ** k) / factorial
    elif scheme == 'gaussian':
        param = param if param is not None else 1.0
        factorial = torch.tensor([np.math.factorial(i) for i in range(num_hops+1)])
        return (param ** torch.arange(num_hops+1)) / factorial
    elif scheme == 'uniform':
        param = param if param is not None else np.sqrt(3/(num_hops+1))
        theta = torch.rand(num_hops+1) * 2 * param - param
        return theta/torch.norm(theta, p=1)
    elif scheme == 'normal':
        param = param if param is not None else 1.0
        theta = torch.randn(num_hops+1) * param
        return theta/torch.norm(theta, p=1)
    elif scheme == 'custom':
        return Tensor(param).float()
    else:
        raise NotImplementedError()


class DecoupledFixed(BaseNN):
    r"""Decoupled structure without matrix transformation during propagation.
        Fixed scalar propagation parameters.
    NOTE: Apply conv every forward() call. Not to be mixed with :class:`Precomputed` models.

    Args:
        theta_scheme (str): Method to generate decoupled parameters.
        theta_param (float, optional): Hyperparameter for the scheme.
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
        theta_scheme = kwargs.pop('theta_scheme', 'appr')
        theta_param = kwargs.pop('theta_param', 0.5)
        theta = gen_theta(num_hops, theta_scheme, theta_param)

        conv_cls = load_import(conv, lib)
        # NOTE: k=0 layer explicitly handles x without propagation. So there is
        # (num_hops+1) conv layers in total.
        return nn.ModuleList([
            conv_cls(num_hops=num_hops, hop=k, theta=theta[k], **kwargs) for k in range(num_hops+1)])


class DecoupledVar(BaseNN):
    r"""Decoupled structure without matrix transformation during propagation.
        Learnable scalar propagation parameters.
    NOTE: Apply conv every forward() call. Not to be mixed with :class:`Precomputed` models.

    Args:
        theta_scheme (str): Method to generate decoupled parameters.
        theta_param (float, optional): Hyperparameter for the scheme.
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
        theta_scheme = kwargs.pop('theta_scheme', 'ones')
        theta_param = kwargs.pop('theta_param', 1.0)
        self.theta_init = gen_theta(num_hops, theta_scheme, theta_param)

        conv_cls = load_import(conv, lib)
        # NOTE: k=0 layer explicitly handles x without propagation. So there is
        # (num_hops+1) conv layers in total.
        return nn.ModuleList([
            conv_cls(num_hops=num_hops, hop=k, theta=nn.Parameter(self.theta_init[k]), **kwargs) for k in range(num_hops+1)])

    def reset_parameters(self):
        if self.in_layers > 0:
            self.in_mlp.reset_parameters()
        if self.out_layers > 0:
            self.out_mlp.reset_parameters()
        for k, conv in enumerate(self.convs):
            conv.reset_parameters()
            conv.theta.data = self.theta_init[k].clone()

# FEATURE: DecoupledCpu class
# TODO: PrecomputedFixed class inherited from DecoupledFixed
