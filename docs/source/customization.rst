Customization
=============

Add New Spectral Filter
-----------------------

New spectral filters to :mod:`pyg_spectral.nn.conv` can be easily implemented by **only three steps**, then enjoys a range of model architectures, analysis utilities, and training schemes.

Step 1: Define propagation matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The base class :class:`pyg_spectral.nn.conv.base_mp.BaseMP` provides essential methods for building spectral filters. We can define a new filter class :class:`pyg_spectral.nn.conv.SkipConv` by inheriting from it:

.. code-block:: python

    from torch import Tensor
    from pyg_spectral.nn.conv.base_mp import BaseMP

    class SkipConv(BaseMP):
        def __init__(self, num_hops, hop, cached, **kwargs):
            kwargs['propagate_mat'] = 'A-I'
            super(SkipConv, self).__init__(num_hops, hop, cached, **kwargs)

The propagation matrix is specified by the :obj:`propagate_mat` argument as a string. Each matrix can be the normalized adjacency matrix (:obj:`A`) or the normalized Laplacian matrix (:obj:`L`), with optional diagonal scaling, where the scaling factor can either be a number or an attribute name of the class. Multiple propagation matrices can be combined by `,`. Valid examples: :obj:`A`, :obj:`L-2*I`, :obj:`L,A+I,L-alpha*I`.

Step 2: Prepare representation matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Similar to ``PyG`` modules, our spectral filter class takes the graph attribute :obj:`x` and edge index :obj:`edge_index` as input. The :meth:`pyg_spectral.nn.conv.base_mp.BaseMP._get_convolute_mat` method prepares the representation matrices used in recurrent computation as a dictionary:

.. code-block:: python

    def _get_convolute_mat(self, x, edge_index):
        return {'x': x, 'x_1': x}

The above example overwrites the method for :class:`SkipConv`, returning the input feature :obj:`x` and a placeholder :obj:`x_1` for the representation in the previous hop.

Step 3: Derive recurrent forward
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :meth:`pyg_spectral.nn.conv.base_mp.BaseMP._forward` method implements recurrent computation of the filter. Its input/output is a dictionary combining the propagation matrices defined by :obj:`propagate_mat` and the representation matrices prepared by :meth:`pyg_spectral.nn.conv.base_mp.BaseMP._get_convolute_mat`.

.. code-block:: python

    def _forward(self, x, x_1, prop):
        if self.hop == 0:
            # No propagation for k=0
            return {'x': x, 'x_1': x, 'prop': prop}

        h = self.propagate(prop, x=x)
        h = h + x_1
        return {'x': h, 'x_1': x, 'prop': prop}

Similar to ``PyG`` modules, the :func:`propagate` method conducts graph propagation by the given matrices. The above example corresponds to the graph propagation with a skip connection to the previous representation: :math:`H^{(k)} = (A-I)H^{(k-1)} + H^{(k-2)}`.

Build the model!
~~~~~~~~~~~~~~~~

Now the :class:`SkipConv` filter is properly defined. The following snippet use the :class:`pyg_spectral.nn.models.DecoupledVar` model composing 10 hops of :class:`SkipConv` filters, which can be used as a normal PyTorch model:

.. code-block:: python

    from pyg_spectral.nn.models import DecoupledVar

    model = DecoupledVar(conv='SkipConv', num_hops=10, in_channels=x.size(1), hidden_channels=x.size(1), out_channels=x.size(1))
    out = model(x, edge_index)


Configure Experiment Parameters
-------------------------------

Refer to the help text by:

.. code-block:: bash

    python benchmark/run_single.py --help

.. code-block::

    usage: python run_single.py
    options:
        --help                      show this help message and exit
        # Logging configuration
        --seed SEED                 random seed
        --dev DEV                   GPU id
        --suffix SUFFIX             Save name suffix.
        -quiet                      Dry run without saving logs.
        --storage {state_file,state_ram,state_gpu}
                                    Storage scheme for saving the checkpoints.
        --loglevel LOGLEVEL         10:progress, 15:train, 20:info, 25:result
        # Data configuration
        --data DATA                 Dataset name
        --data_split DATA_SPLIT     Index or percentage of dataset split
        --normg NORMG               Generalized graph norm
        --normf [NORMF]             Embedding norm dimension. 0: feat-wise, 1: node-wise, None: disable
        # Model configuration
        --model MODEL               Model class name
        --conv CONV                 Conv class name
        --num_hops NUM_HOPS         Number of conv hops
        --in_layers IN_LAYERS       Number of MLP layers before conv
        --out_layers OUT_LAYERS     Number of MLP layers after conv
        --hidden HIDDEN             Number of hidden width
        --dp_lin  DP_LIN            Dropout rate for linear
        --dp_conv DP_CONV           Dropout rate for conv
        # Training configuration
        --epoch EPOCH               Number of epochs
        --patience PATIENCE         Patience epoch for early stopping
        --period PERIOD             Periodic saving epoch interval
        --batch BATCH               Batch size
        --lr_lin  LR_LIN            Learning rate for linear
        --lr_conv LR_CONV           Learning rate for conv
        --wd_lin  WD_LIN            Weight decay for linear
        --wd_conv WD_CONV           Weight decay for conv
        # Model-specific
        --theta_scheme THETA_SCHEME Filter name
        --theta_param THETA_PARAM   Hyperparameter for filter
        --combine {sum,sum_weighted,cat}
                                    How to combine different channels of convs
        # Conv-specific
        --alpha ALPHA               Decay factor
        --beta BETA                 Scaling factor
        # Test flags
        --test_deg                  Call TrnFullbatch.test_deg()

Add New Experiment Dataset
--------------------------

In ``benchmark/trainer/load_data.py``, append the :meth:`SingleGraphLoader._resolve_import` method to include new datasets under respective protocols.
