# Spectral-GNN-Benchmark

## Installation

For development
```bash
pip install -e .
```

Create your own branch for development and pull request.

## Experiment
```bash
cd examples
bash scripts/sfb.sh
```

## Structure

### `examples`
Related to benchmark experiments.

### `spectral_gnn`
Core codes for spectral GNNs following the PyTorch Geometric structure.

### `log`
Log files for saving the results. For experiment and ignored by git.

### `data`
PyG dataset path. For experiment and ignored by git. Link to:
- `/Resource/dataset/PyG/` on Dell7920
- `/share/data/dataset/PyG/` on triangle-001
