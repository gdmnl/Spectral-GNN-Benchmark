# Benchmarking Spectral Graph Neural Networks

`pyg_spectral` is a [PyTorch Geometric](https://pyg.org)-based framework for analyzing, implementing, and benchmarking spectral GNNs with effectiveness and efficiency evaluations.

## Installation

This package can be easily installed by [pip](https://pip.pypa.io/en/stable/):

```bash
pip install -e .
```

The installation script already covers the following dependencies:
* [PyTorch](https://github.com/pytorch/pytorch) (`>=2.0`): please follow the [official guide](https://pytorch.org/get-started/locally/) if a specific CUDA version is required.
* [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric) (`>=2.5.3`)
* [TorchMetrics](https://github.com/Lightning-AI/torchmetrics) (`>=1.0`): only required for `examples/` experiments.
* [Optuna](https://github.com/optuna/optuna) (`>=3.4`): only required for hyperparameter search in `examples/` experiments.

## Reproduce Experiments
### Main Experitmet
Effectiveness and efficiency of spectral GNNs.
Datasets will be automatically downloaded and processed by the code.

Script for full-batch models (*Table 2, 8, 9*):
```bash
cd examples
bash scripts/runfb.sh
```

Script for mini-batch models (*Table 3, 10, 11*):
```bash
bash scripts/runmb.sh
```

### Effect of Graph Normalization
Graph Normalization vs degree-speficic accuracy (*Table 9*):

```bash
bash scripts/eval_degree.sh
```

### Effect of Propagation Hop
Number of propagation hop vs Accuracy (*Table 7, 8*):

```bash
bash scripts/eval_hop.sh
```

## Customization
### Configure Experiment Parameters
Refer to help text by:
```bash
python examples/run_single.py --help
```

### Add New Experiment Dataset
In `examples/trainer/load_data.py`, edit the `SingleGraphLoader._resolve_import()` method to include new datasets under respective protocols.

### Add New Spectral Filter


## Framework Arrangement

### Covered Models

| **Category** | **Model** |
|:------------:|:----------|
| Fixed Filter | [GCN](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html), |

### Covered Datasets

| **Source** | **Graph** |
|:------------:|:----------|
| [PyG](https://pytorch-geometric.readthedocs.io/en/stable/modules/datasets.html) | cora, |
| [OGB]() |  |
| [LINKX](https://github.com/CUAI/Non-Homophily-Large-Scale) |  |
| [Yandex]() |  |

### Code Structure

* `examples/`: codes for benchmark experiments.
* `pyg_spectral/`: core codes for spectral GNNs designs, arranged in [PyG](https://github.com/pyg-team/pytorch_geometric) structure.
  * `nn.conv`: spectral spectral filters, similar to [`torch_geometric.nn.conv`](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#convolutional-layers).
  * `nn.models`: common neural network architectures, similar to [`torch_geometric.nn.models`](https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#models).
* `log/`: experiment log files and parameter search results.
* `data/`: raw and processed datasets arranged following different protocals.

![Code structure of this framework and relation to PyG.](docs/code_struct.png)

## Roadmap
* [ ] Support C++ propagation backend with efficient algorithms.
* [ ] Support more transformation operations.
* [ ] Support iterative eigen-decomposition for full-spectrum spectral filters.

## Misc
* This project is licensed under the [MIT LICENSE](LICENSE).

* Please refer to the [CONTRIBUTING](docs/CONTRIBUTING.md) guide for contributing to this project.

* Use the "Cite this repository" on the right column for [CITATION](docs/CITATION.md)
