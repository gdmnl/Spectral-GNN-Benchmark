Installation
----------------------------------------

This package can be easily installed by running `pip <https://pip.pypa.io/en/stable/>`__ at package root path:

.. code-block:: bash

   pip install -r requirements.txt
   pip install -e .[benchmark]

The installation script already covers the following core dependencies:

- `PyTorch <https://github.com/pytorch/pytorch>`__ (``>=2.0`` [1]_)
- `PyTorch Geometric <https://github.com/pyg-team/pytorch_geometric>`__ (``>=2.5.3``)
- `TorchMetrics <https://github.com/Lightning-AI/torchmetrics>`__ (``>=1.0``): only required for ``benchmark/`` experiments.
- `Optuna <https://github.com/optuna/optuna>`__ (``>=3.4``): only required for hyperparameter search in ``benchmark/`` experiments.


Advanced Options
++++++++++++++++++++++++

Installation can be specified by pip options. The following options can also be combined on demand.

Only ``pyg_spectral`` Package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install without any options:

.. code-block:: bash

   pip install -e .

Benchmark Experiments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install with ``[benchmark]`` option:

.. code-block:: bash

   pip install -e .[benchmark]

Docs Development
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install with ``[docs]`` option:

.. code-block:: bash

   pip install -e .[docs]

C++ Backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Ensure C++ 11 is installed.

.. code-block:: bash

   gcc --version

2. Install with ``[cpp]`` option and environment variable ``PSFLAG_CPP=1``:

.. code-block:: bash

   export PSFLAG_CPP=1; pip install -e .[cpp]

.. [1] Please refer to the `official guide <https://pytorch.org/get-started/locally/>`__ if a specific CUDA version is required for PyTorch.
