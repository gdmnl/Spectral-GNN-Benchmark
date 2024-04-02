from typing import Tuple, Union, Final

import torch
from torch import Tensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import spmm

from pyg_spectral.nn.conv.fix_sumadj import gen_theta
from pyg_spectral.utils import dropout_edge


class VarSumAdj(MessagePassing):
    r"""Summation of hops of adj-based propagation with learnable parameters.
    Similar to [GPRGNN](https://github.com/jianhao2016/GPRGNN/blob/master/src/GNN_models.py)

    Covers:
        - 'appr': GPRGNN

    Args:
        theta: spectral filter:
            1. (Tensor): Custom list of hop parameters.
            2. (Tuple[str, float]): Method to generate parameters.
            3. (Tuple[str, List[float]]): Custom list of hop parameters.
        K (int, optional): Number of iterations :math:`K`. If K=0 then infer from p.
        dropedge (float, optional): Edge dropout. Defaults to 0.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F)`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F)`
    """
    supports_edge_weight: bool = True
    supports_batch: bool = False
    supports_decouple: bool = True
    supports_norm_batch: bool = False
    # supports_edge_attr: Final[bool] = False   # Always False

    def __init__(self, theta: Union[Tensor, Tuple[str, float]] = ('appr', 0.1),
                 K: int = 0, dropedge: float = 0., **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(VarSumAdj, self).__init__(**kwargs)

        if isinstance(theta, tuple):
            self.scheme = theta[0]
            theta = gen_theta(K, *theta)
        else:
            self.scheme = 'custom'
        self.theta_init = theta
        self.theta = torch.nn.Parameter(theta)
        self.K = K if K > 0 else len(theta)
        assert len(theta) >= self.K, f'Hop K={self.K} larger than {len(self.theta)} hop parameters!'
        self.dropedge = dropedge

        if self.__class__ == VarSumAdj:
            self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.theta.data = self.theta_init

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
        return f'{self.__class__.__name__}-{self.scheme}(K={self.K}, alpha={self.theta[0]:f})'


class VarLinSumAdj(VarSumAdj):
    r"""Summation of hops of adj-based propagation with learnable parameters,
    then perform linear transformation.

    Covers:

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:**
          node features :math:`(|\mathcal{V}|, F_{out})`
    """
    supports_edge_weight: Final[bool] = True
    supports_batch: Final[bool] = True
    supports_decouple: Final[bool] = False
    supports_norm_batch: Final[bool] = False

    def __init__(self, in_channels: int, out_channels: int,
                 theta: Union[Tensor, Tuple[str, float]] = ('appr', 0.1),
                 K: int = 0, dropedge: float = 0., bias: bool = True, **kwargs):
        super(VarLinSumAdj, self).__init__(theta, K, dropedge, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin = Linear(in_channels, out_channels, bias=bias)

        if self.__class__ == VarLinSumAdj:
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
        return (f'{self.__class__.__name__}-{self.scheme}({self.in_channels}, '
                f'{self.out_channels}, K={self.K}, alpha={self.theta[0]:f})')
