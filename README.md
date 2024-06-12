# Benchmarking Spectral Graph Neural Networks

`pyg_spectral` is a [PyTorch Geometric](https://pyg.org)-based framework for analyzing, implementing, and benchmarking spectral GNNs with effectiveness and efficiency evaluations.

> [!IMPORTANT]
> ***Why this project?***  
> We list the following highlights of our framework compared to PyG and similar works:
> - **Unified Framework**: We offer a plug-and-play collection for spectral models and filters in unified and efficient implementations, rather than a model-specific design. Our rich collection greatly extends the PyG model zoo.
> - **Spectral-oriented Design**: We decouple non-spectral designs and feature the pivotal spectral kernel being consistent throughout different settings. Most filters are thus easily adaptable to a wide range of model-level options, including those provided by PyG and PyG-based frameworks.
> - **High scalability**: As spectral GNNs are inherently suitable for large-scale learning, our framework is feasible to common scalable learning schemes and acceleration techniques. Several spectral-oriented approximation algorithms are also supported.

---

- [Installation](#installation)
- [Reproduce Experiments](#reproduce-experiments)
  - [Main Experiments](#main-experiments)
  - [Additional Experiments](#additional-experiments)
- [Customization](#customization)
  - [Configure Experiment Parameters](#configure-experiment-parameters)
  - [Add New Experiment Dataset](#add-new-experiment-dataset)
  - [Add New Spectral Filter](#add-new-spectral-filter)
- [Framework Arrangement](#framework-arrangement)
  - [Covered Models](#covered-models)
  - [Covered Datasets](#covered-datasets)
  - [Code Structure](#code-structure)
- [Roadmap](#roadmap)

## Installation

This package can be easily installed by [pip](https://pip.pypa.io/en/stable/). Execute the following two commands in the root path of this repository:
```bash
pip install -r requirements.txt
pip install -e .
```

The installation script already covers the following core dependencies:
- [PyTorch](https://github.com/pytorch/pytorch) (`>=2.0`[^1])
- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) (`>=2.5.3`[^2])
- [TorchMetrics](https://github.com/Lightning-AI/torchmetrics) (`>=1.0`): only required for `benchmark/` experiments.
- [Optuna](https://github.com/optuna/optuna) (`>=3.4`): only required for hyperparameter search in `benchmark/` experiments.

[^1]: Please refer to the [official guide](https://pytorch.org/get-started/locally/) if a specific CUDA version is required for PyTorch.
[^2]: Please refer to the [official guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) if a specific CUDA version is required for PyG.

For additional installation of the C++ backend, please refer to [propagations/README.md](pyg_spectral/propagations/README.md#installation).

## Reproduce Experiments
### Main Experiments
Acquire results on the effectiveness and efficiency of spectral GNNs.
Datasets will be automatically downloaded and processed by the code.

#### Run full-batch models (*Table 2, 8, 9*):
```bash
cd benchmark
bash scripts/runfb.sh
```

#### Run mini-batch models (*Table 3, 10, 11*):
```bash
bash scripts/runmb.sh
```

### Additional Experiments
#### Effect of graph normalization vs degree-speficic accuracy (*Figure 3, 9*):
```bash
bash scripts/eval_degree.sh
```

Figures can be plotted by: [`benchmark/notebook/fig_degng.ipynb`](benchmark/notebook/fig_degng.ipynb).

#### Effect of the number of propagation hops vs accuracy (*Figure 7, 8*):
```bash
bash scripts/eval_hop.sh
```

Figures can be plotted by: [`benchmark/notebook/fig_hop.ipynb`](benchmark/notebook/fig_hop.ipynb).

#### Frequency response ():

## Customization
### Configure Experiment Parameters
Refer to the help text by:
```bash
python benchmark/run_single.py --help
```
```
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
```

### Add New Experiment Dataset
In `benchmark/trainer/load_data.py`, append the `SingleGraphLoader._resolve_import()` method to include new datasets under respective protocols.

### Add New Spectral Filter
New spectral filters to `pyg_spectral/nn/conv/` can be easily implemented by **only three steps**, then enjoys a range of model architectures, analysis utilities, and training schemes.

#### Step 1: Define propagation matrix
The base class `BaseMP` provides essential methods for building spectral filters. We can define a new filter class `SkipConv` by inheriting from it:
```python
from torch import Tensor
from pyg_spectral.nn.conv.base_mp import BaseMP

class SkipConv(BaseMP):
    def __init__(self, num_hops, hop, cached, **kwargs):
        kwargs['propagate_mat'] = 'A-I'
        super(SkipConv, self).__init__(num_hops, hop, cached, **kwargs)
```

The propagation matrix is specified by the `propagate_mat` argument as a string. Each matrix can be the normalized adjacency matrix (`A`) or the normalized Laplacian matrix (`L`), with optional diagonal scaling, where the scaling factor can either be a number or an attribute name of the class. Multiple propagation matrices can be combined by `,`. Valid examples: `A`, `L-2*I`, `L,A+I,L-alpha*I`.

#### Step 2: Prepare representation matrix
Similar to PyG modules, our spectral filter class takes the graph attribute `x` and edge index `edge_index` as input. The `_get_convolute_mat()` method prepares the representation matrices used in recurrent computation as a dictionary:
```python
def _get_convolute_mat(self, x, edge_index):
    return {'x': x, 'x_1': x}
```

The above example overwrites the method for `SkipConv`, returning the input feature `x` and a placeholder `x_1` for the representation in the previous hop.

#### Step 3: Derive recurrent forward
The `_forward()` method implements recurrent computation of the filter. Its input/output is a dictionary combining the propagation matrices defined by `propagate_mat` and the representation matrices prepared by `_get_convolute_mat()`. 
```python
def _forward(self, x, x_1, prop):
    if self.hop == 0:
        # No propagation for k=0
        return {'x': x, 'x_1': x, 'prop': prop}

    h = self.propagate(prop, x=x)
    h = h + x_1
    return {'x': h, 'x_1': x, 'prop': prop}
```

Similar to PyG modules, the `propagate()` method conducts graph propagation by the given matrices. The above example corresponds to the graph propagation with a skip connection to the previous representation: $H^{(k)} = (A-I)H^{(k-1)} + H^{(k-2)}$.

#### Build the model!
Now the `SkipConv` filter is properly defined. The following snippet use the `DecoupledVar` model composing 10 hops of `SkipConv` filters, which can be used as a normal PyTorch model:
```python
from pyg_spectral.nn.models import DecoupledVar

model = DecoupledVar(conv='SkipConv', num_hops=10, in_channels=x.size(1), hidden_channels=x.size(1), out_channels=x.size(1))
out = model(x, edge_index)
```

## Framework Arrangement

### Covered Models

| **Category** | **Model** |
|:------------:|:----------|
| Fixed Filter | [GCN](https://arxiv.org/abs/1609.02907), [SGC](https://arxiv.org/pdf/1902.07153), [gfNN](https://arxiv.org/pdf/1905.09550), [GZoom](https://arxiv.org/pdf/1910.02370), [S$^2$GC](https://openreview.net/pdf?id=CYO5T-YjWZV),[GLP](https://arxiv.org/pdf/1901.09993), [APPNP](https://arxiv.org/pdf/1810.05997), [GCNII](https://arxiv.org/pdf/2007.02133), [GDC](https://proceedings.neurips.cc/paper_files/paper/2019/file/23c894276a2c5a16470e6a31f4618d73-Paper.pdf), [DGC](https://arxiv.org/pdf/2102.10739), [AGP](https://arxiv.org/pdf/2106.03058), [GRAND+](https://arxiv.org/pdf/2203.06389)|
|Variable Filter|[GIN](https://arxiv.org/pdf/1810.00826), [AKGNN](https://arxiv.org/pdf/2112.04575), [DAGNN](https://dl.acm.org/doi/pdf/10.1145/3394486.3403076), [GPRGNN](https://arxiv.org/pdf/2006.07988), [ARMAGNN](https://arxiv.org/pdf/1901.01343), [ChebNet](https://papers.nips.cc/paper_files/paper/2016/file/04df4d434d481c5bb723be1b6df1ee65-Paper.pdf), [ChebNetII](https://arxiv.org/pdf/2202.03580), [HornerGCN/ClenshawGCN](https://arxiv.org/pdf/2210.16508), [BernNet](https://arxiv.org/pdf/2106.10994), [LegendreNet](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10160025), [JacobiConv](https://arxiv.org/pdf/2205.11172), [FavardGNN/OptBasisGNN](https://arxiv.org/pdf/2302.12432)|
|Filter Bank|[AdaGNN](https://arxiv.org/pdf/2104.12840), [FBGNN](https://arxiv.org/pdf/2008.08844), [ACMGNN](https://arxiv.org/pdf/2210.07606), [FAGCN](https://arxiv.org/pdf/2101.00797), [G$^2$CN](https://proceedings.mlr.press/v162/li22h/li22h.pdf), [GNN-LF/HF](https://arxiv.org/pdf/2101.11859), [FiGURe](https://arxiv.org/pdf/2310.01892)|


### Covered Datasets

| **Source** | **Graph** |
|:------------:|:----------|
| [PyG](https://pytorch-geometric.readthedocs.io/en/stable/modules/datasets.html) | cora, citeseer, pubmed, flickr, actor |
| [OGB](https://ogb.stanford.edu/docs/nodeprop/) | ogbn-arxiv, ogbn-mag, ogbn-products |
| [LINKX](https://github.com/CUAI/Non-Homophily-Large-Scale) | penn94, arxiv-year, genius, twitch-gamer, snap-patients, pokec, wiki |
| [Yandex](https://github.com/yandex-research/heterophilous-graphs) | chameleon, squirrel, roman-empire, minesweeper, amazon-ratings, questions, tolokers |

### Code Structure

- `benchmark/`: codes for benchmark experiments.
- `pyg_spectral/`: core codes for spectral GNNs designs, arranged in [PyG](https://github.com/pyg-team/pytorch_geometric) structure.
  - `nn.conv`: spectral spectral filters, similar to [`torch_geometric.nn.conv`](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#convolutional-layers).
  - `nn.models`: common neural network architectures, similar to [`torch_geometric.nn.models`](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#models).
  - `nn.propagations`: C++ backend for efficient propagation algorithms.
- `log/`: experiment log files and parameter search results.
- `data/`: raw and processed datasets arranged following different protocals.

![Code structure of this framework and relation to PyG.](docs/code_struct.png)

## Roadmap
- [ ] Support C++ propagation backend with efficient algorithms.
  - [x] Unifews
  - [ ] SGC
  - [ ] GBP, AGP
- [ ] Support more transformation operations.
  - [ ] Generalize ACMGNN
- [ ] Support iterative eigen-decomposition for full-spectrum spectral filters.
  - [ ] Jacobi method
  - [ ] Lanczos method

## Misc
- This project is licensed under the [MIT LICENSE](LICENSE).
- Use "Cite this repository" in the right sidebar for [CITATION](docs/CITATION.cff)
<!-- - Please refer to the [CONTRIBUTING](docs/CONTRIBUTING.md) guide for contributing to this project. -->
