from typing import Tuple, Union

import numpy as np
import torch
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import spmm, dropout_edge


def gen_theta(K: int, scheme: str, alpha: float = None) -> Tensor:
    r"""Generate list of hop parameters based on given scheme.

    Args:
        K (int): Number of hops.
        scheme (str): Method to generate parameters.
            - `khop`: K-hop, :math:`\mathbf{H} = \mathbf{A}^K \mathbf{X}`.
            - 'appr': Approximate PPR, :math:`\mathbf{H} = \sum_{k=0}^K \alpha (1 - \alpha)^k \mathbf{A}^k \mathbf{X}`.
            - 'nappr': Negative PPR, :math:`\mathbf{H} = \sum_{k=0}^K \alpha^k \mathbf{A}^k \mathbf{X}`.
            - `mono`: Monomial, :math:`\mathbf{H} = \frac{1}{K} \sum_{k=1}^K\left((1-\alpha)\mathbf{A}^k + \alpha \mathbf{I}\right) \mathbf{X}`.
            - 'hk': Heat Kernel, :math:`\mathbf{H} = \sum_{k=0}^K e^{-\alpha k} \mathbf{A}^k \mathbf{X}`.
            - 'uniform': Random uniform distribution.
            - 'gaussian': Random Gaussian distribution.
        alpha (float, optional): Hyperparameter for the scheme.
            - 'khop': Only the alpha-hop, :math:`\alpha \in [0, K]`.
            - 'appr': Decay factor, :math:`\alpha \in [0, 1]`.
            - 'nappr': Decay factor, :math:`\alpha \in [-1, 1]`.
            - 'mono': Decay factor, :math:`\alpha \in [0, 1]`.
            - 'hk': Decay factor, :math:`\alpha > 0`.
            - 'uniform': Distribution bound.
            - 'gaussian': Distribution variance.

    Returns:
        theta (Tensor): List of hop parameters.
    """
    assert K > 0, 'K should be a positive integer'
    if scheme == 'khop':
        alpha = alpha if alpha is not None else K
        theta = torch.zeros(K+1)
        theta[alpha] = 1
        return theta
    elif scheme == 'appr':
        alpha = alpha if alpha is not None else 0.5
        # theta[-1] = (1 - alpha) ** K
        return alpha * (1 - alpha) ** torch.arange(K+1)
    elif scheme == 'nappr':
        alpha = alpha if alpha is not None else 0.5
        theta = alpha ** torch.arange(K+1)
        return theta/torch.norm(theta, p=1)
    elif scheme == 'mono':
        alpha = alpha if alpha is not None else 0.5
        theta = torch.zeros(K+1)
        theta[0] = alpha
        theta[1:] = (1 - alpha) / K
        return theta
    elif scheme == 'hk':
        alpha = alpha if alpha is not None else 1.0
        return torch.exp(-alpha * torch.arange(K+1))
    elif scheme == 'uniform':
        alpha = alpha if alpha is not None else np.sqrt(3/(K+1))
        theta = torch.rand(K+1) * 2 * alpha - alpha
        return theta/torch.norm(theta, p=1)
    elif scheme == 'gaussian':
        alpha = alpha if alpha is not None else 1.0
        theta = torch.randn(K+1) * alpha
        return theta/torch.norm(theta, p=1)
    else:
        raise NotImplementedError()


class FixSumAdj(MessagePassing):
    r"""Summation of hops of adj-based propagations with fixed parameters.
    Similar to :class:`pyg.nn.conv.APPNP`

    Covers:
        - `khop`: SGC, DAGNN
        - 'appr': APPNP
        - 'mono': S2GC
        - 'hk': GDC, AGP

    Args:
        theta (Tensor): List of hop parameters.
        K (int, optional): Number of iterations :math:`K`. If K=0 then infer from p.
        dropedge (float, optional): Edge dropout. Defaults to 0.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F)`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F)`
    """
    def __init__(self, theta: Union[Tensor, Tuple[str, float]] = ('appr', 0.1),
                 K: int = 0, dropedge: float = 0., **kwargs):
        # FEATURE: `combine_root` as `pyg.nn.conv.SimpleConv`
        kwargs.setdefault('aggr', 'add')
        super(FixSumAdj, self).__init__(**kwargs)

        if isinstance(theta, tuple):
            theta = gen_theta(K, *theta)
        self.theta = theta
        self.K = K if K > 0 else len(theta)
        self.dropedge = dropedge

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
    ) -> Tensor:

        if self.theta[0] == 0:
            h = torch.zeros_like(x)
        else:
            h = x * self.theta[0]
        for k in range(1, self.K+1):
            # Edge dropout
            if self.dropedge > 0:
                assert isinstance(edge_index, Tensor) and \
                    edge_index.size(0) == 2, \
                    'DropEdge only supports tensor format edge_index'
                edge_index, edge_mask = dropout_edge(
                    edge_index, p=self.dropedge, training=self.training)
                if edge_weight is not None:
                    edge_weight = edge_weight[edge_mask]

            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight)
            if self.theta[k] != 0:
                h += x * self.theta[k]

        return h

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(K={self.K}, alpha={self.theta[0]:f})'


class FixLinSumAdj(FixSumAdj):
    r"""Summation of hops of adj-based propagations with fixed parameters, then
    perform linear transformation. Similar to :class:`pyg.nn.conv.SSGConv`

    Covers:
        * `khop`: SGC, DAGNN
        * 'mono': S2GC

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:**
          node features :math:`(|\mathcal{V}|, F_{out})`
    """
    def __init__(self, in_channels: int, out_channels: int,
                 theta: Union[Tensor, Tuple[str, float]] = ('appr', 0.1),
                 K: int = 0, dropedge: float = 0., bias: bool = True, **kwargs):
        super(FixLinSumAdj, self).__init__(theta, K, dropedge, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = Linear(in_channels, out_channels, bias=bias)

        if self.__class__ == FixLinSumAdj:
            self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
    ) -> Tensor:

        h = super().forward(x, edge_index, edge_weight)
        return self.lin(h)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={self.K}, alpha={self.theta[0]:f})')
