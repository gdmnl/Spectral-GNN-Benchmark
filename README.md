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
bash scripts/pfb_var_s.sh
```

## Structure

### `examples`
Related to benchmark experiments.

### `pyg_spectral`
Core codes for spectral GNNs mimicking PyTorch Geometric structure.

### `log`
Log files for saving the results. For experiment and ignored by git. Link to:
- `/share/data/transfer/Spectral-GNN-Benchmark/log` on triangle-001

### `data`
PyG dataset path. For experiment and ignored by git. Link to:
- `/Resource/dataset/PyG/` on Dell7920
- `/share/data/dataset/PyG/` on triangle-001
