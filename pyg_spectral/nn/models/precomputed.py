from typing import Optional

import torch
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from pyg_spectral.nn.models.decoupled import DecoupledFixed, DecoupledVar, DecoupledFixedCompose, DecoupledVarCompose


class PrecomputedFixed(DecoupledFixed):
    r"""Decoupled structure with precomputation separating propagation from transformation.
        Fixed scalar propagation parameters and accumulating precompute results.
    NOTE: Only apply propagation in `convolute()`. Not to be mixed with :class:`Decoupled` models.

    Args:
        theta_scheme (str): Method to generate decoupled parameters.
        theta_param (float, optional): Hyperparameter for the scheme.
        --- BaseNN Args ---
        conv (str): Name of :class:`pyg_spectral.nn.conv` module.
        num_hops (int): Total number of conv hops.
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
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

    def __init__(self, in_layers: Optional[int] = None, **kwargs):
        assert in_layers is None or in_layers == 0, "PrecomputedFixed does not support in_layers."
        super(PrecomputedFixed, self).__init__(in_layers=in_layers, **kwargs)

    def convolute(self,
        x: Tensor,
        edge_index: Adj,
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


class PrecomputedVar(DecoupledVar):
    r"""Decoupled structure with precomputation separating propagation from transformation.
        Learnable scalar propagation parameters and storing all intermediate precompute results.
    NOTE: Only apply propagation in `convolute()`. Not to be mixed with :class:`Decoupled` models.

    Args:
        theta_scheme (str): Method to generate decoupled parameters.
        theta_param (float, optional): Hyperparameter for the scheme.
        --- BaseNN Args ---
        conv (str): Name of :class:`pyg_spectral.nn.conv` module.
        num_hops (int): Total number of conv hops.
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
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
    def __init__(self, in_layers: Optional[int] = None, **kwargs):
        assert in_layers is None or in_layers == 0, "PrecomputedVar does not support in_layers."
        super(PrecomputedVar, self).__init__(in_layers=in_layers, **kwargs)

    def convolute(self,
        x: Tensor,
        edge_index: Adj,
    ) -> list:
        r"""Decoupled propagation step for calling the convolutional module.
            `self._forward()` should not contain derivable computations.
        Returns:
            embed (Tensor): List of precomputed node embeddings of each hop.
                Each shape is :math:`(|\mathcal{V}|, F, len(convs)+1)`.
        """
        conv_mat = self.get_forward_mat(x, edge_index, comp_scheme='convolute')

        xi = conv_mat['x'] if 'x' in conv_mat else conv_mat['out']
        xs = [xi]
        for conv in self.convs:
            conv.comp_scheme = 'convolute'
            conv_mat = conv._forward(**conv_mat)
            xi = conv_mat['x'] if 'x' in conv_mat else conv_mat['out']
            xs.append(xi)

        return torch.stack(xs, dim=xs[0].dim())

    def forward(self,
        xs: Tensor,
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
        conv_mat = self.get_forward_mat(xs[..., 0], None, comp_scheme='forward')
        for k, conv in enumerate(self.convs):
            conv.comp_scheme = 'forward'
            key = 'x' if 'x' in conv._forward.__code__.co_varnames else 'out'
            conv_mat[key] = xs[..., k+1]
            conv_mat = conv(**conv_mat)
        out = conv_mat['out']
        if self.out_layers > 0:
            out = self.out_mlp(out, batch=batch, batch_size=batch_size)
        return out


# ==========
class PrecomputedFixedCompose(DecoupledFixedCompose):
    r"""Decoupled structure with precomputation separating propagation from transformation.
        Fixed scalar propagation parameters and accumulating precompute results.

    Args:
        theta_scheme (List[str]): Method to generate decoupled parameters.
        theta_param (List[float], optional): Hyperparameter for the scheme.
        combine (str): How to combine different channels of convs. (one of
            "sum", "sum_weighted", "cat").
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

    def convolute(self,
        x: Tensor,
        edge_index: Adj,
    ) -> Tensor:
        r"""Decoupled propagation step for calling the convolutional module.
            Requires no variable transformation in conv.forward().
        Returns:
            embed (Tensor): Precomputed node embeddings.
                Shape: :math:`(|\mathcal{V}|, F, Q)`.
        """
        out = []
        conv_mats = self.get_forward_mat()
        for i, channel in enumerate(self.convs):
            conv_mat = conv_mats[i](x, edge_index)
            for conv in channel:
                conv_mat = conv(**conv_mat)
            out.append(conv_mat['out'])
        return torch.stack(out, dim=-1)

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
        out = None
        for i, channel in enumerate(self.convs):
            if i == 0:
                out = x[..., i]
            else:
                if self.combine == 'sum':
                    out = out + x[..., i]
                elif self.combine in ['sum_weighted', 'sum_vec']:
                    out = out + self.gamma[i] * x[..., i]
                elif self.combine == 'cat':
                    out = torch.cat((out, x[..., i]), dim=-1)

        if self.out_layers > 0:
            out = self.out_mlp(out, batch=batch, batch_size=batch_size)
        return out


class PrecomputedVarCompose(DecoupledVarCompose):
    r"""Decoupled structure with precomputation separating propagation from transformation.
        Learnable scalar propagation parameters and storing all intermediate precompute results.

    Args:
        theta_scheme (List[str]): Method to generate decoupled parameters.
        theta_param (List[float], optional): Hyperparameter for the scheme.
        combine (str): How to combine different channels of convs. (one of
            "sum", "sum_weighted", "cat").
        --- BaseNN Args ---
        conv (List[str]): Name of :class:`pyg_spectral.nn.conv` module.
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

    def convolute(self,
        x: Tensor,
        edge_index: Adj,
    ) -> Tensor:
        r"""Decoupled propagation step for calling the convolutional module.
            Requires no variable transformation in conv.forward().
        Returns:
            embed (Tensor): List of precomputed node embeddings of each hop.
                Shape: :math:`(|\mathcal{V}|, F, Q, len(convs)+1)`.
        """
        out = []
        conv_mats = self.get_forward_mat()
        for i, channel in enumerate(self.convs):
            conv_mat = conv_mats[i](x, edge_index, comp_scheme='convolute')

            xi = conv_mat['x'] if 'x' in conv_mat else conv_mat['out']
            xs = [xi]
            for conv in channel:
                conv.comp_scheme = 'convolute'
                conv_mat = conv._forward(**conv_mat)
                xi = conv_mat['x'] if 'x' in conv_mat else conv_mat['out']
                xs.append(xi)

            out.append(torch.stack(xs, dim=xs[0].dim()))
        return torch.stack(out, dim=-2)

    def forward(self,
        xs: Tensor,
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
        out = None
        conv_mats = self.get_forward_mat()

        for i, channel in enumerate(self.convs):
            conv_mat = conv_mats[i](xs[..., i, 0], None, comp_scheme='forward')
            for k, conv in enumerate(channel):
                conv.comp_scheme = 'forward'
                key = 'x' if 'x' in conv._forward.__code__.co_varnames else 'out'
                conv_mat[key] = xs[..., i, k+1]
                conv_mat = conv(**conv_mat)

            if i == 0:
                out = conv_mat['out']
            else:
                if self.combine == 'sum':
                    out = out + conv_mat['out']
                elif self.combine in ['sum_weighted', 'sum_vec']:
                    out = out + self.gamma[i] * conv_mat['out']
                elif self.combine == 'cat':
                    out = torch.cat((out, conv_mat['out']), dim=-1)

        if self.out_layers > 0:
            out = self.out_mlp(out, batch=batch, batch_size=batch_size)
        return out
