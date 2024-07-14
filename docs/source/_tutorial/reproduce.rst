Reproduce Experiments
----------------------------------------

Main Experiments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Acquire results on the effectiveness and efficiency of spectral GNNs.
Datasets will be automatically downloaded and processed by the code.

**Run full-batch models** (*Table 2, 8, 9*):

.. code-block:: console

   $ cd benchmark
   $ bash scripts/runfb.sh

**Run mini-batch models** (*Table 3, 10, 11*):

.. code-block:: console

   $ bash scripts/runmb.sh

Additional Experiments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Effect of graph normalization** (*Figure 3, 9*):

.. code-block:: console

   $ bash scripts/eval_degree.sh

Figures can be plotted by: `benchmark/notebook/fig_degng.ipynb <https://github.com/gdmnl/Spectral-GNN-Benchmark/blob/main/benchmark/notebook/fig_degng.ipynb>`_.

**Effect of propagation hops** (*Figure 7, 8*):

.. code-block:: console

   $ bash scripts/eval_hop.sh

Figures can be plotted by: `benchmark/notebook/fig_hop.ipynb <https://github.com/gdmnl/Spectral-GNN-Benchmark/blob/main/benchmark/notebook/fig_hop.ipynb>`_.

**Frequency response** (*Table 12*):

.. code-block:: console

   $ bash scripts/exp_filter.sh
