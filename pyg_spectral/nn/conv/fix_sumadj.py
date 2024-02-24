import torch
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import spmm, dropout_edge


def gen_theta(K: int, scheme: str, decay: float = None) -> Tensor:
    r"""Generate list of hop parameters based on given scheme.

    Args:
        K (int): Number of iterations :math:`K`.
        scheme (str): Method to generate parameters.
            * `sgc`: only K-hop, :math:`\mathbf{H} = \mathbf{A}^K \mathbf{X}`.
            * 'appnp': Personalized PageRank, :math:`\mathbf{H} = \sum_{k=0}^K \alpha (1 - \alpha)^k \mathbf{A}^k \mathbf{X}`.
            * `ssgc`: Monomial, :math:`\mathbf{H} = \frac{1}{K} \sum_{k=1}^K\left((1-\alpha)\mathbf{A}^k + \alpha \mathbf{I}\right) \mathbf{X}`.
            * 'hk': Heat Kernel, :math:`\mathbf{H} = \sum_{k=0}^K e^{-\alpha k} \mathbf{A}^k \mathbf{X}`.
        decay: Decay factor among hops.
            * 'sgc': not used.
            * 'appnp': :math:`decay \in [0, 1]`.
            * 'ssgc'::math:`decay \in [0, 1]`.
            * 'hk': :math:`decay > 0`.

    Returns:
        theta (Tensor): List of hop parameters.
    """
    assert K > 1, 'K should be greater than 1'
    if scheme == 'sgc':
        return torch.tensor([0.0] * (K - 1) + [1.0])
    elif scheme == 'appnp':
        decay = decay if decay is not None else 0.5
        return decay * (1 - decay) ** torch.arange(K)
    elif scheme == 'ssgc':
        decay = decay if decay is not None else 0.5
        theta = torch.zeros(K)
        theta[0] = decay
        theta[1:] = (1 - decay) / (K - 1)
        return theta
    elif scheme == 'hk':
        decay = decay if decay is not None else 1.0
        return torch.exp(-decay * torch.arange(K))
    else:
        raise NotImplementedError()


class FixSumAdj(MessagePassing):
    r"""Summation of hops of adj-based propagations with fixed parameters.
    Similar to `pyg.nn.conv.APPNP`

    Args:
        theta (Tensor): List of hop parameters.
        K (int, optional): Number of iterations :math:`K`. If K=0 then infer from p.
        dropout (float, optional): Edge dropout. Defaults to 0.
    """
    def __init__(self, theta: Tensor, K: int = 0,
                 dropout: float = 0., **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.theta = theta
        self.K = K if K > 0 else len(theta)
        self.dropout = dropout

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
        for k in range(1, self.K):
            # Edge dropout
            edge_index, edge_mask = dropout_edge(edge_index, p=self.dropout, training=self.training)
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
        return f'{self.__class__.__name__}(K={self.K}, decay={self.theta[0]:f})'


class FixLinSumAdj(MessagePassing):
    r"""Summation of hops of adj-based propagations with fixed parameters, then
    perform linear transformation. Similar to `pyg.nn.conv.SSGConv`

    Args:
        theta (Tensor): List of hop parameters.
        K (int, optional): Number of iterations :math:`K`. If K=0 then infer from p.
        dropout (float, optional): Edge dropout. Defaults to 0.
    """
    def __init__(self, in_channels: int, out_channels: int,
                 theta: Tensor, K: int = 0,
                 dropout: float = 0., bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.theta = theta
        self.K = K if K > 0 else len(theta)
        self.dropout = dropout

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = Linear(in_channels, out_channels, bias=bias)
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

        if self.theta[0] == 0:
            h = torch.zeros_like(x)
        else:
            h = x * self.theta[0]
        for k in range(1, self.K):
            # Edge dropout
            edge_index, edge_mask = dropout_edge(edge_index, p=self.dropout, training=self.training)
            if edge_weight is not None:
                edge_weight = edge_weight[edge_mask]

            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight)
            if self.theta[k] != 0:
                h += x * self.theta[k]

        return self.lin(h)

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={self.K}, decay={self.theta[0]:f})')
