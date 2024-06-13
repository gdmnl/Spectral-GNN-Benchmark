.. pyg_spectral documentation master file, created by
   sphinx-quickstart on Thu Jun 13 13:17:07 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pyg_spectral
========================================
``pyg_spectral`` is a `PyTorch Geometric <https://pyg.org>`_-based framework for analyzing, implementing, and benchmarking spectral GNNs with effectiveness and efficiency evaluations.

*Why this project?*
   We list the following highlights of our framework compared to PyG and similar works:

   * **Unified Framework**: We offer a plug-and-play collection for spectral models and filters in unified and efficient implementations, rather than a model-specific design. Our rich collection greatly extends the PyG model zoo.

   * **Spectral-oriented Design**: We decouple non-spectral designs and feature the pivotal spectral kernel being consistent throughout different settings. Most filters are thus easily adaptable to a wide range of model-level options, including those provided by PyG and PyG-based frameworks.

   * **High scalability**: As spectral GNNs are inherently suitable for large-scale learning, our framework is feasible to common scalable learning schemes and acceleration techniques. Several spectral-oriented approximation algorithms are also supported.

Installation
----------------------------------------
This package can be easily installed by running `pip <https://pip.pypa.io/en/stable/>`_ at package root path:

.. code-block:: bash

   pip install -r requirements.txt
   pip install -e .

The installation script already covers the following core dependencies:

- `PyTorch <https://github.com/pytorch/pytorch>`_ (``>=2.0`` [1]_)
- `PyTorch Geometric <https://github.com/pyg-team/pytorch_geometric>`_ (``>=2.5.3``)
- `TorchMetrics <https://github.com/Lightning-AI/torchmetrics>`_ (``>=1.0``): only required for ``benchmark/`` experiments.
- `Optuna <https://github.com/optuna/optuna>`_ (``>=3.4``): only required for hyperparameter search in ``benchmark/`` experiments.

.. [1] Please refer to the `official guide <https://pytorch.org/get-started/locally/>`_ if a specific CUDA version is required for PyTorch.

Reproduce Experiments
----------------------------------------

Main Experiments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Acquire results on the effectiveness and efficiency of spectral GNNs.
Datasets will be automatically downloaded and processed by the code.

**Run full-batch models** (*Table 2, 8, 9*):

.. code-block:: bash

   cd benchmark
   bash scripts/runfb.sh

**Run mini-batch models** (*Table 3, 10, 11*):

.. code-block:: bash

   bash scripts/runmb.sh

Additional Experiments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Effect of graph normalization vs degree-specific accuracy** (*Figure 3, 9*):

.. code-block:: bash

   bash scripts/eval_degree.sh

Figures can be plotted by: `benchmark/notebook/fig_degng.ipynb <https://github.com/gdmnl/Spectral-GNN-Benchmark/blob/main/benchmark/notebook/fig_degng.ipynb>`_.

**Effect of the number of propagation hops vs accuracy** (*Figure 7, 8*):

.. code-block:: bash

   bash scripts/eval_hop.sh

Figures can be plotted by: `benchmark/notebook/fig_hop.ipynb <https://github.com/gdmnl/Spectral-GNN-Benchmark/blob/main/benchmark/notebook/fig_hop.ipynb>`_.

**Frequency response** (*Table 12*):

.. code-block:: bash

   bash scripts/exp_filter.sh


.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   customization
   arrangement

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Package Reference

   _include/pyg_spectral.nn
   _include/pyg_spectral.nn.conv
   _include/pyg_spectral.nn.models
   _include/pyg_spectral.profile
   _include/pyg_spectral.transforms
   _include/pyg_spectral.utils
