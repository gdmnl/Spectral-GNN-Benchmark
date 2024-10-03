Configure Benchmark
===============================

Experiment Parameters
-------------------------------

Refer to the help text by:

.. code-block:: console

    $ python benchmark/run_single.py --help

--help                      show this help message and exit

.. rubric:: Logging configuration

--seed SEED                 random seed
--dev DEV                   GPU id
--suffix SUFFIX             Result log file name. ``None``:not saving results
-quiet                      File log. ``True``:dry run without saving logs
--storage STORAGE
                            Checkpoint log storage scheme.
                            Options: ``state_file``, ``state_ram``, ``state_gpu``
--loglevel LOGLEVEL         Console log. ``10``:progress, ``15``:train, ``20``:info, ``25``:result

.. rubric:: Data configuration

--data DATA                 Dataset name
--data_split DATA_SPLIT     Index or percentage of dataset split
--normg NORMG               Generalized graph norm
--normf NORMF               Embedding norm dimension. ``0``: feat-wise, ``1``: node-wise, ``None``: disable

.. rubric:: Model configuration

--model MODEL               Model class name
--conv CONV                 Conv class name
--num_hops NUM_HOPS         Number of conv hops
--in_layers IN_LAYERS       Number of MLP layers before conv
--out_layers OUT_LAYERS     Number of MLP layers after conv
--hidden_channels HIDDEN    Number of hidden width
--dropout_lin DP_LIN        Dropout rate for linear
--dropout_conv DP_CONV      Dropout rate for conv

.. rubric:: Training configuration

--epoch EPOCH               Number of epochs
--patience PATIENCE         Patience epoch for early stopping
--period PERIOD             Periodic saving epoch interval
--batch BATCH               Batch size
--lr_lin LR_LIN             Learning rate for linear
--lr_conv LR_CONV           Learning rate for conv
--wd_lin WD_LIN             Weight decay for linear
--wd_conv WD_CONV           Weight decay for conv

.. rubric:: Model-specific

--theta_scheme THETA_SCHEME  Filter name
--theta_param THETA_PARAM   Hyperparameter for filter
--combine COMBINE
                            How to combine different channels of convs.
                            Options: ``sum``, ``sum_weighted``, ``cat``

.. rubric:: Conv-specific

--alpha ALPHA               Decay factor
--beta BETA                 Scaling factor

.. rubric:: Test flags

--test_deg                  Call :meth:`test_deg() <benchmark.trainer.TrnFullbatch.test_deg()>`

Add New Dataset
--------------------------

Append the :meth:`SingleGraphLoader._resolve_import() <benchmark.trainer.SingleGraphLoader._resolve_import()>` method to include new datasets under respective protocols.
