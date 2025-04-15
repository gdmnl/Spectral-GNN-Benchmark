from .yandex import Yandex
from .linkx import LINKX, FB100
from .grid2d import Grid2D

from .utils import (
    split_crossval,
    T_insert, resolve_data, resolve_split
)

import importlib
dataset_map, f_get_data = {}, {}
for classi in ['yandex', 'linkx', 'ogbl', 'ogbn', 'pygn']:
    module = importlib.import_module(f".{classi}", __name__)
    f_get_data[classi] = module.get_data
    for datai in module.DATA_LIST:
        assert datai not in dataset_map, f"Duplicate dataset: {datai}"
        dataset_map[datai] = classi

__all__ = [
    'Yandex', 'LINKX', 'FB100', 'Grid2D',
    'T_insert', 'resolve_data', 'resolve_split',
]
