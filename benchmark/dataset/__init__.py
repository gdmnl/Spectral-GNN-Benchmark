from .yandex import Yandex
from .linkx import LINKX, FB100
from .grid2d import Grid2D

from .utils import (
    split_random,
    T_insert, resolve_data, resolve_split
)

import importlib
class_list, func_list = {}, {}
for classi in ['yandex', 'linkx', 'ogbn', 'pyg']:
    module = importlib.import_module(f".{classi}", __name__)
    func_list[classi] = module.get_data
    for datai in module.DATA_LIST:
        assert datai not in class_list, f"Duplicate dataset: {datai}"
        class_list[datai] = classi

__all__ = [
    'Yandex', 'LINKX', 'FB100', 'Grid2D',
    'T_insert', 'resolve_data', 'resolve_split',
]
