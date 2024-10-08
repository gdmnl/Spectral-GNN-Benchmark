.. pyg_spectral documentation master file, created by
   sphinx-quickstart on Thu Jun 13 13:17:07 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pyg_spectral
========================================
.. sidebar:: Project links

   | 🔍 `Documentation <https://gdmnl.github.io/Spectral-GNN-Benchmark/>`__
   | 👾 `GitHub <https://github.com/gdmnl/Spectral-GNN-Benchmark/>`__ |gh-bn|
   | 📄 `Paper <https://arxiv.org/abs/2406.09675>`__
   | 📎 `Citation <https://github.com/gdmnl/Spectral-GNN-Benchmark#misc>`__

``pyg_spectral`` is a `PyG <https://pyg.org>`_-based framework for analyzing, implementing, and benchmarking spectral GNNs with effectiveness and efficiency evaluations.

.. admonition:: *Why this project?*

   We list the following highlights of our framework compared to PyG and similar works:

   * **Unified Framework**: We offer a plug-and-play collection for spectral models and filters in unified and efficient implementations, rather than a model-specific design. Our rich collection greatly extends the PyG model zoo.
   * **Spectral-oriented Design**: We decouple non-spectral designs and feature the pivotal spectral kernel being consistent throughout different settings. Most filters are thus easily adaptable to a wide range of model-level options, including those provided by PyG and PyG-based frameworks.
   * **High scalability**: As spectral GNNs are inherently suitable for large-scale learning, our framework is feasible to common scalable learning schemes and acceleration techniques. Several spectral-oriented approximation algorithms are also supported.

.. include:: _tutorial/installation.rst
   :end-before: Advanced Options

For advanced options, please refer to `Installation Options <_tutorial/installation.html#advanced-options>`_.

.. include:: _tutorial/reproduce.rst

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   _tutorial/installation
   _tutorial/reproduce
   _tutorial/customize
   _tutorial/configure
   _tutorial/arrangement

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

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Experiments

   _include/benchmark.trainer
   _include/benchmark.utils
   _include/benchmark.dataset

.. [1] Please refer to the `official guide <https://pytorch.org/get-started/locally/>`__ if a specific CUDA version is required for PyTorch.

.. |gh-bn| raw:: html

   <script async defer src="https://buttons.github.io/buttons.js"></script>
   <a class="github-button" href="https://github.com/gdmnl/Spectral-GNN-Benchmark" vertical-align="middle" data-show-count="true" aria-label="Star gdmnl/Spectral-GNN-Benchmark on GitHub">Star</a>
