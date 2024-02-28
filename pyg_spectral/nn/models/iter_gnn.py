from typing import Final

import torch.nn as nn
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.nn.conv import MessagePassing

from pyg_spectral.utils import load_import


class IterConv(BasicGNN):
    r"""Use module from :class:`pyg_spectral.nn.conv` as convolutional layer.

    Args:
        conv (str): Name of :class:`pyg_spectral.nn.conv` module.
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        lib_conv (str, optional): Parent module library other than
            :class:`pyg_spectral.nn.conv`.
        **kwargs (optional): Additional arguments of the
            :class:`pyg_spectral.nn.conv` module.
    """
    supports_edge_weight: Final[bool] = True
    supports_edge_attr: Final[bool] = False
    supports_norm_batch: Final[bool]

    def init_conv(self, in_channels: int, out_channels: int,
                  conv: str, **kwargs) -> MessagePassing:
        lib = kwargs.pop('lib_conv', 'pyg_spectral.nn.conv')
        return load_import(conv, lib)(in_channels, out_channels, **kwargs)

    def _get_name(self) -> str:
        return super().__repr__()

    def __repr__(self) -> str:
        return super(BasicGNN, self).__repr__()
