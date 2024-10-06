from typing import Any, Callable

import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import ChebConv


class ChebNet(nn.Module):
    name = 'DecoupledFixed'
    conv_name = lambda _: 'ChebConv'
    pargs = ['conv', 'num_hops', 'dropout_lin',
             'in_channels', 'hidden_channels', 'out_channels',]
    param = {
        'num_hops':         ('int', (2, 30), {'step': 2}, lambda x: x),
        'hidden_channels':  ('categorical', ([16, 32, 64, 128, 256],), {}, lambda x: x),
        'dropout_lin':      ('float', (0.0, 1.0), {'step': 0.1}, lambda x: round(x, 2)),
    }

    def __init__(self,
            conv: str,
            num_hops: int = 0,
            in_channels: int | None = None,
            hidden_channels: int | None = None,
            out_channels: int | None = None,
            dropout_lin: float = 0.,
            act: str | Callable | None = "relu",
            act_first: bool = False,
            act_kwargs: dict[str, Any | None] = None,
            norm: str | Callable | None = "batch_norm",
            norm_kwargs: dict[str, Any | None] = None,
            bias: bool | list[bool] = True,
            **kwargs):
        super().__init__()
        self.conv1 = ChebConv(
            in_channels, hidden_channels,
            K=num_hops//2,
            bias=bias)
        self.conv2 = ChebConv(
            hidden_channels, out_channels,
            K=num_hops//2,
            bias=bias)
        self.dropout_lin = dropout_lin

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout_lin, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
