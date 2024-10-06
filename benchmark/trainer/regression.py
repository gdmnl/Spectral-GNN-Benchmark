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
from .load_data import DATAPATH
from dataset import split_crossval
from utils import ResLogger


class TrnRegression(TrnFullbatch):
    name: str = 'regression'

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


class RegressionLoader(object):
    r"""Loader for regression learning datas.
    """
    def __init__(self, args: Namespace, res_logger: ResLogger = None) -> None:
        r"""Assigning dataset identity.
        """
        self.seed = args.seed
        self.data = args.data.lower()
        self.logger = logging.getLogger('log')
        self.res_logger = res_logger or ResLogger()
        self.metric = None
        self.in_channels = 1
        self.out_channels = 1
        self.transform = T.Compose([
            T.ToSparseTensor(remove_edge_index=True, layout=torch.sparse_csr),  # torch.sparse.Tensor
        ])

    # ===== Data acquisition
    def _resolve_import(self, args: Namespace) -> tuple[str, str, dict]:
        assert self.data in ['2dgrid']
        module_name = 'dataset'
        class_name = 'Grid2D'
        kwargs = dict(
            root=DATAPATH.joinpath('Grid2D'),
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
        data.y = torch.tensor(np.load(DATAPATH.joinpath(f'Grid2D/y_{args.filter_type}.npy')), dtype=torch.float)
        args.in_channels, args.out_channels = self.in_channels, self.out_channels
        args.metric = self.metric

        (r_train, r_val) = map(int, args.data_split.split('/')[:2])
        r_train, r_val = r_train / 100, r_val / 100
        train_mask, val_mask, test_mask = split_crossval(data.y[:,args.img_idx], r_train, r_val)
        data.train_mask = torch.as_tensor(train_mask)
        data.val_mask = torch.as_tensor(val_mask)
        data.test_mask = torch.as_tensor(test_mask)

        self.logger.info(f"[dataset]: {dataset} (features={self.in_channels}, classes={self.out_channels})")
        self.logger.info(f"[data]: {data}")
        self.logger.info(f"[metric]: {metric}")
        self.res_logger.concat([('data', self.data, str), ('metric', metric, str)])
        del dataset
        return data, metric

    def __call__(self, *args, **kwargs):
        return self.get(*args, **kwargs)

    def __str__(self) -> str:
        return f"{self.data}({self.metric})"
