from typing import Any, Callable, Dict, List, Optional, Union

from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from pyg_spectral.nn.models.decoupled import DecoupledFixed, DecoupledVar


class PrecomputedFixed(DecoupledFixed):
    def __init__(self,
            conv: str,
            num_hops: int = 0,
            in_channels: Optional[int] = None,
            hidden_channels: Optional[int] = None,
            out_channels: Optional[int] = None,
            in_layers: Optional[int] = None,
            out_layers: Optional[int] = None,
            dropout_lin: Union[float, List[float]] = 0.,
            dropout_conv: float = 0.,
            act: Union[str, Callable, None] = "relu",
            act_first: bool = False,
            act_kwargs: Optional[Dict[str, Any]] = None,
            norm: Union[str, Callable, None] = "batch_norm",
            norm_kwargs: Optional[Dict[str, Any]] = None,
            plain_last: bool = False,
            bias: Union[bool, List[bool]] = True,
            **kwargs):
        assert in_layers is None or in_layers == 0, "PrecomputedFixed does not support in_layers."
        super(PrecomputedFixed, self).__init__(
            conv=conv,
            num_hops=num_hops,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            in_layers=in_layers,
            out_layers=out_layers,
            dropout_lin=dropout_lin,
            dropout_conv=dropout_conv,
            act=act,
            act_first=act_first,
            act_kwargs=act_kwargs,
            norm=norm,
            norm_kwargs=norm_kwargs,
            plain_last=plain_last,
            bias=bias,
            **kwargs
        )

    def convolute(self,
        x: Tensor,
        edge_index: Adj,
        batch: OptTensor = None,
        batch_size: Optional[int] = None,
    ) -> Tensor:
        r"""Decoupled propagation step for calling the convolutional module.
            Requires no variable transformation in conv.forward().
        Returns:
            embed (Tensor): Precomputed node embeddings.
        """
        conv_mat = self.get_forward_mat(x, edge_index)
        for conv in self.convs:
            conv_mat = conv(**conv_mat)
        return conv_mat['out']

    def forward(self,
        x: Tensor,
        batch: OptTensor = None,
        batch_size: Optional[int] = None,
    ) -> Tensor:
        r"""
        Args:
            x (Tensor): the output `embed` from `convolute()`.
            batch (Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example.
                Only needs to be passed in case the underlying normalization
                layers require the :obj:`batch` information.
                (default: :obj:`None`)
            batch_size (int, optional): The number of examples :math:`B`.
                Automatically calculated if not given.
                Only needs to be passed in case the underlying normalization
                layers require the :obj:`batch` information.
                (default: :obj:`None`)
        """
        if self.out_layers > 0:
            x = self.out_mlp(x, batch=batch, batch_size=batch_size)
        return x

