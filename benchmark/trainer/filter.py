from typing import Tuple
from argparse import Namespace
import logging

import numpy as np
import torch
import torch.nn as nn
from torchmetrics.regression import R2Score, MeanAbsoluteError

from torch_geometric.data import Data, Dataset
import torch_geometric.transforms as T
from pyg_spectral.utils import load_import

from .fullbatch import TrnFullbatch
from .load_metric import ResCollection
from .load_data import DATAPATH, split_random
from utils import ResLogger


class TrnFilter(TrnFullbatch):
    name: str = 'filter'

    def __init__(self,
                 model: nn.Module,
                 data: Dataset,
                 args: Namespace,
                 **kwargs):
        super(TrnFullbatch, self).__init__(model, data, args, **kwargs)
        metric = ResCollection({
            's_r2': R2Score(),
            's_mae': MeanAbsoluteError(),
        }).to(self.device)
        self.evaluator = {k: metric.clone(postfix='_'+k) for k in self.splits}
        self.criterion = nn.MSELoss()

        self.mask: dict = None
        self.img_idx = args.img_idx
        self.flag_test_deg = args.test_deg if hasattr(args, 'test_deg') else False

    def _fetch_input(self) -> tuple:
        input, label = (self.data.x[:, self.img_idx:self.img_idx+1], self.data.adj_t), self.data.y[:, self.img_idx:self.img_idx+1]
        if hasattr(self.model, 'preprocess'):
            self.model.preprocess(*input)
        return input, label


class FilterLoader(object):
    r"""Loader for filter learning datas.
    """
    def __init__(self, args: Namespace, res_logger: ResLogger = None) -> None:
        r"""Assigning dataset identity.
        """
        self.seed = args.seed
        self.data = args.data.lower()
        self.logger = logging.getLogger('log')
        self.res_logger = res_logger or ResLogger()
        self.metric = None
        self.num_features = 1
        self.num_classes = 1
        self.transform = T.Compose([
            T.ToSparseTensor(remove_edge_index=True, layout=torch.sparse_csr),  # torch.sparse.Tensor
        ])

    # ===== Data acquisition
    def _resolve_import(self, args: Namespace) -> Tuple[str, str, dict]:
        assert self.data in ['2dgrid']
        module_name = 'dataset_process'
        class_name = 'Filter'
        kwargs = dict(
            root=DATAPATH.joinpath('Filter'),
            name=self.data,
            transform=self.transform)
        self.metric = 's_r2'

        kwargs['root'] = kwargs['root'].resolve().absolute()
        return module_name, class_name, kwargs, self.metric

    def get(self, args: Namespace) -> Data:
        r"""Load data based on parameters.
        """

        self.logger.debug('-'*20 + f" Loading data: {self} " + '-'*20)

        module_name, class_name, kwargs, metric = self._resolve_import(args)

        dataset = load_import(class_name, module_name)(**kwargs)
        data = dataset[0]

        # get specific filtered graph signal.
        data.y = torch.tensor(np.load(DATAPATH.joinpath(f'Filter/y_{args.filter_type}.npy')), dtype=torch.float)
        args.num_features, args.num_classes = self.num_features, self.num_classes
        args.metric = self.metric

        (r_train, r_val) = map(int, args.data_split.split('/')[:2])
        r_train, r_val = r_train / 100, r_val / 100
        train_mask, val_mask, test_mask = split_random(data.y[:,args.img_idx], r_train, r_val)
        data.train_mask = torch.as_tensor(train_mask)
        data.val_mask = torch.as_tensor(val_mask)
        data.test_mask = torch.as_tensor(test_mask)

        self.logger.info(f"[dataset]: {dataset} (features={self.num_features}, classes={self.num_classes})")
        self.logger.info(f"[data]: {data}")
        self.logger.info(f"[metric]: {metric}")
        self.res_logger.concat([('data', self.data, str), ('metric', metric, str)])
        del dataset
        return data, metric

    def __call__(self, *args, **kwargs):
        return self.get(*args, **kwargs)

    def __str__(self) -> str:
        return f"{self.data}({self.metric})"
