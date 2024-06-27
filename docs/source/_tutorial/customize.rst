Customize Spectral Modules
==================================

Add New Filter
-----------------------

New spectral filters to :mod:`pyg_spectral.nn.conv` can be easily implemented by **only three steps**, then enjoys a range of model architectures, analysis utilities, and training schemes.

Step 1: Define propagation matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The base class :class:`nn.conv.BaseMP <pyg_spectral.nn.conv.BaseMP>` provides essential methods for building spectral filters. We can define a new filter class :class:`nn.conv.SkipConv` by inheriting from it:

.. code-block:: python

    from torch import Tensor
    from pyg_spectral.nn.conv.base_mp import BaseMP

    class SkipConv(BaseMP):
        def __init__(self, num_hops, hop, cached, **kwargs):
            kwargs['propagate_mat'] = 'A-I'
            super(SkipConv, self).__init__(num_hops, hop, cached, **kwargs)

The propagation matrix is specified by the :obj:`propagate_mat` argument as a string. Each matrix can be the normalized adjacency matrix (``A``) or the normalized Laplacian matrix (``L``), with optional diagonal scaling, where the scaling factor can either be a number or an attribute name of the class. Multiple propagation matrices can be combined by ``,``. Valid examples: ``A``, ``L-2*I``, ``L,A+I,L-alpha*I``.

Step 2: Prepare representation matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Similar to PyG modules, our spectral filter class takes the graph attribute :obj:`x` and edge index :obj:`edge_index` as input. The :meth:`_get_convolute_mat() <pyg_spectral.nn.conv.BaseMP._get_convolute_mat()>` method prepares the representation matrices used in recurrent computation as a dictionary:

.. code-block:: python

        def _get_convolute_mat(self, x, edge_index):
            return {'x': x, 'x_1': x}

The above example overwrites the method for :class:`SkipConv`, returning the input feature :obj:`x` and a placeholder :obj:`x_1` for the representation in the previous hop.

Step 3: Derive recurrent forward
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :meth:`_forward() <pyg_spectral.nn.conv.BaseMP._forward()>` method implements recurrent computation of the filter. Its input/output is a dictionary combining the propagation matrices defined by :obj:`propagate_mat` and the representation matrices prepared by :meth:`_get_convolute_mat() <pyg_spectral.nn.conv.BaseMP._get_convolute_mat()>`.

.. code-block:: python

        def _forward(self, x, x_1, prop):
            if self.hop == 0:
                # No propagation for k=0
                return {'x': x, 'x_1': x, 'prop': prop}

            h = self.propagate(prop, x=x)
            h = h + x_1
            return {'x': h, 'x_1': x, 'prop': prop}

Similar to PyG modules, the :meth:`propagate() <torch_geometric.nn.conv.MessagePassing.propagate>` method conducts graph propagation by the given matrices. The above example corresponds to the graph propagation with a skip connection to the previous representation: :math:`H^{(k)} = (A-I)H^{(k-1)} + H^{(k-2)}`.

Build the model!
~~~~~~~~~~~~~~~~

Now the :class:`SkipConv` filter is properly defined. The following snippet use the :class:`nn.models.DecoupledVar <pyg_spectral.nn.models.DecoupledVar>` model composing 10 hops of :class:`SkipConv` filters, which can be used as a normal PyTorch model:

.. code-block:: python

    from pyg_spectral.nn.models import DecoupledVar

    model = DecoupledVar(conv='SkipConv', num_hops=10, in_channels=x.size(1), hidden_channels=x.size(1), out_channels=x.size(1))
    out = model(x, edge_index)
