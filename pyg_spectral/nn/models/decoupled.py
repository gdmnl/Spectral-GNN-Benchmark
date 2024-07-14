from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset

from pyg_spectral.nn.models.base_nn import BaseNN, BaseNNCompose
from pyg_spectral.utils import load_import


def gen_theta(num_hops: int, scheme: str, param: Union[float, List[float]] = None) -> Tensor:
    r"""Generate list of hop parameters based on given scheme.

    Args:
        num_hops (int): Total number of hops.
        scheme (str): Method to generate parameters.
            - `zeros`: all-zero, :math:`\theta_k = 0`.
            - `ones`: all-same, :math:`\theta_k = p`.
            - `impulse`: K-hop, :math:`\theta_K = p, else 0`.
            - `inverse`: Inverse, :math:`\theta_k = p/(k+1)`.
            - `mono`: Monomial, :math:`\theta_k = (1-p)/K, \theta_0 = p`.
            - 'appr': Approximate PPR, :math:`\theta_k = p (1 - p)^k`.
            - 'nappr': Negative PPR, :math:`\theta_k = p^k`.
            - 'hk': Heat Kernel, :math:`\theta_k = e^{-p}p^k / k!`.
            - 'gaussian': Graph Gaussian Kernel, :math:`theta_k = p^{k} / k!`.
            - 'chebyshev': Chebyshev polynomial.
            - 'uniform': Random uniform distribution.
            - 'normal_std': Standard random Gaussian distribution N(0, p).
            - 'normal': Random Gaussian distribution N(p0, p1).
            - 'custom': Custom list of hop parameters.
        param (float, optional): Hyperparameter for the scheme.
            - `zeros`: NA.
            - `ones`: Value.
            - 'impulse': Value.
            - 'inverse': Value.
            - 'mono': Decay factor, :math:`p \in [0, 1]`.
            - 'appr': Decay factor, :math:`p \in [0, 1]`.
            - 'nappr': Decay factor, :math:`p \in [-1, 1]`.
            - 'hk': Decay factor, :math:`p > 0`.
            - 'gaussian': Decay factor, :math:`p > 0`.
            - 'chebyshev': NA.
            - 'uniform': Distribution bound.
            - 'normal_std': Distribution variance.
            - 'normal': Distribution mean and variance.
            - 'custom': Float list of hop parameters.

    Returns:
        theta (Tensor): Lenth (num_hops+1) list of hop parameters.
    """
    assert num_hops > 0, 'num_hops should be a positive integer'
    if scheme == 'zeros':
        return torch.zeros(num_hops+1, dtype=torch.float)
    elif scheme == 'ones':
        param = param if param is not None else 1.0
        return torch.ones(num_hops+1, dtype=torch.float) * param
    elif scheme == 'impulse':
        param = param if param is not None else 1.0
        theta = torch.zeros(num_hops+1, dtype=torch.float)
        theta[num_hops] = param
        return theta
    elif scheme == 'inverse':
        param = param if param is not None else 1.0
        return torch.tensor([param/(k+1) for k in range(num_hops+1)], dtype=torch.float)
    elif scheme == 'mono':
        param = param if param is not None else 0.5
        theta = torch.zeros(num_hops+1, dtype=torch.float)
        theta[0] = param
        theta[1:] = (1.0 - param) / num_hops
        return theta
    elif scheme == 'appr':
        param = param if param is not None else 0.5
        # theta[-1] = (1 - param) ** num_hops
        return param * (1.0 - param) ** torch.arange(num_hops+1)
    elif scheme == 'nappr':
        param = param if param is not None else 0.5
        theta = param ** torch.arange(num_hops+1)
        return theta/torch.norm(theta, p=1)
    elif scheme == 'hk':
        param = param if param is not None else 1.0
        k = torch.arange(num_hops+1)
        factorial = torch.tensor([np.math.factorial(i) for i in range(num_hops+1)])
        return torch.exp(-torch.tensor(param, dtype=torch.float)) * (param ** k) / factorial
    elif scheme == 'gaussian':
        param = param if param is not None else 1.0
        factorial = torch.tensor([np.math.factorial(i) for i in range(num_hops+1)])
        return (param ** torch.arange(num_hops+1)) / factorial
    elif scheme == 'chebyshev':
        return (torch.cos((num_hops-torch.arange(num_hops+1)+0.5) * torch.pi/(num_hops+1))) ** 2
    elif scheme == 'uniform':
        param = param if param is not None else np.sqrt(3/(num_hops+1))
        theta = torch.rand(num_hops+1) * 2 * param - param
        return theta/torch.norm(theta, p=1)
    elif scheme == 'normal_std':
        param = param if param is not None else 1.0
        theta = torch.randn(num_hops+1) * param
        return theta/torch.norm(theta, p=1)
    elif scheme == 'normal':
        param = param if param is not None else [0.0, 1.0]
        if isinstance(param, float):
            param = [param, 1.0]
        elif len(param) == 1:
            param.append(1.0)
        return torch.normal(param[0], param[1], size=(num_hops+1,))
    elif scheme == 'custom':
        return torch.tensor(param, dtype=torch.float)
    else:
        raise NotImplementedError()


class DecoupledFixed(BaseNN):
    r"""Decoupled structure without matrix transformation during propagation.
    Fixed scalar propagation parameters.

    .. Note ::
        Apply conv every :meth:`forward()` call.
        Not to be mixed with :class:`Precomputed` models.

    Args:
        theta_scheme (str): Method to generate decoupled parameters.
        theta_param (float, optional): Hyperparameter for the scheme.
        conv, num_hops, in_channels, hidden_channels, out_channels:
            args for :class:`BaseNN`
        in_layers, out_layers, dropout_lin, dropout_conv, lib_conv:
            args for :class:`BaseNN`
        act, act_first, act_kwargs, norm, norm_kwargs, plain_last, bias:
            args for :class:`torch_geometric.nn.models.MLP`.
        **kwargs: Additional arguments of :class:`pyg_spectral.nn.conv`.
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
        # NOTE: k=0 layer explicitly handles x without propagation. So there are
        # (num_hops+1) conv layers in total.
        convs = nn.ModuleList([
            conv_cls(num_hops=num_hops, hop=k, **kwargs) for k in range(num_hops+1)])
        for k, convk in enumerate(convs):
            convk.register_buffer('theta', theta[k].clone())
            if hasattr(convk, '_init_with_theta'):
                convk._init_with_theta()
        return convs


class DecoupledVar(BaseNN):
    r"""Decoupled structure without matrix transformation during propagation.
    Learnable scalar propagation parameters.

    .. Note ::
        Apply conv every :meth:`forward()` call.
        Not to be mixed with :class:`Precomputed` models.

    Args:
        theta_scheme (str): Method to generate decoupled parameters.
        theta_param (float, optional): Hyperparameter for the scheme.
        conv, num_hops, in_channels, hidden_channels, out_channels:
            args for :class:`BaseNN`
        in_layers, out_layers, dropout_lin, dropout_conv, lib_conv:
            args for :class:`BaseNN`
        act, act_first, act_kwargs, norm, norm_kwargs, plain_last, bias:
            args for :class:`torch_geometric.nn.models.MLP`.
        **kwargs: Additional arguments of :class:`pyg_spectral.nn.conv`.
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
        # NOTE: k=0 layer explicitly handles x without propagation. So there are
        # (num_hops+1) conv layers in total.
        convs = nn.ModuleList([
            conv_cls(num_hops=num_hops, hop=k, **kwargs) for k in range(num_hops+1)])
        for k, convk in enumerate(convs):
            convk.register_parameter('theta', nn.Parameter(self.theta_init[k].clone()))
            if hasattr(convk, '_init_with_theta'):
                convk._init_with_theta()
        return convs

    def reset_parameters(self):
        if self.in_layers > 0:
            self.in_mlp.reset_parameters()
        if self.out_layers > 0:
            self.out_mlp.reset_parameters()
        for k, conv in enumerate(self.convs):
            reset(conv)
            conv.theta.data = self.theta_init[k].clone()


# ==========
class DecoupledFixedCompose(BaseNNCompose):
    r"""Decoupled structure without matrix transformation during propagation.
    Fixed scalar propagation parameters.

    Args:
        theta_scheme (List[str]): Method to generate decoupled parameters.
        theta_param (List[float], optional): Hyperparameter for the scheme.
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

    def init_conv(self,
        conv: str,
        num_hops: int,
        lib: str,
        **kwargs
    ) -> MessagePassing:
        conv = conv.split(',')
        convs = nn.ModuleList()

        theta_schemes = kwargs.pop('theta_scheme', 'appr')
        theta_schemes = theta_schemes.split(',')
        theta_params = kwargs.pop('theta_param', [0.5])
        if not isinstance(theta_params, list):
            theta_params = [theta_params] * len(conv)

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

    def init_conv(self,
        conv: str,
        num_hops: int,
        lib: str,
        **kwargs
    ) -> MessagePassing:
        conv = conv.split(',')
        convs = nn.ModuleList()
        self.theta_init = []

        theta_schemes = kwargs.pop('theta_scheme', ['ones'])
        theta_schemes = theta_schemes.split(',')
        if len(theta_schemes) == 1:
            theta_schemes = theta_schemes * len(conv)
        theta_params = kwargs.pop('theta_param', [1.0])
        if not isinstance(theta_params, list):
            theta_params = [theta_params]
        if len(theta_params) == 1:
            theta_params = theta_params * len(conv)


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
