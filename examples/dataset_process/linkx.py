import os.path as osp
from typing import Any, Callable, Optional

import numpy as np
import scipy
from sklearn.preprocessing import label_binarize
import pandas as pd
import torch

from torch_geometric.data import Data, InMemoryDataset, download_url, download_google_url
from torch_geometric.utils import coalesce
from torch_geometric.transforms import BaseTransform

from .utils import even_quantile_labels


NCLASS_Q = 5


class LINKX(InMemoryDataset):
    r"""
    paper: Large Scale Learning on Non-Homophilous Graphs: New Benchmarks and Strong Simple Methods
    ref: https://github.com/CUAI/Non-Homophily-Large-Scale/
    """
    dataset_drive_url = {
        'snap-patents.mat' : '1ldh23TSY1PwXia6dU0MYcpyEgX-w3Hia',
        'pokec.mat' : '1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y',
        'yelp-chi.mat': '1fAXtTVQS4CfEk4asqrFw9EPmlUPGbGtJ',
        'twitch-gamer_features.csv' : '1fA9VIIEI8N0L27MSQfcBzJgRQLvSbrvR',
        'twitch-gamer_edges.csv' : '1XLETC6dG3lVl7kDmytEJ52hvDMVdxnZ0',
        'wiki_views.pt': '1p5DlVHrnFgYm3VsNIzahSsvCD424AyvP', # Wiki 1.9M
        'wiki_edges.pt': '14X7FlkjrlUgmnsYtPwdh-gGuFla4yb5u', # Wiki 1.9M
        'wiki_features.pt': '1ySNspxbK-snNoAZM7oxiWGvOnTRdSyEK' # Wiki 1.9M
    }
    splits_drive_url = {
        'snap-patents_splits.npy' : '12xbBRqd8mtG_XkNLH8dRRNZJvVM4Pw-N',
        'pokec_splits.npy' : '1ZhpAiyTNc0cE_hhgyiqxnkKREHK7MK-_',
    }

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.name = name.lower()
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        if self.name in ["twitch-gamer",]:
            return [f'{self.name}_edges.csv', f'{self.name}_features.csv']
        elif self.name in ["wiki"]:
            return [f'{self.name}_edges.pt', f'{self.name}_features.pt', f'{self.name}_views.pt']
        return f'{self.name}.mat'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        if self.name in ['genius']:
            url = "https://github.com/CUAI/Non-Homophily-Large-Scale/raw/master/data"
            download_url(f'{url}/{self.name}.mat', self.raw_dir)
            return

        for fname in self.dataset_drive_url:
            if fname.startswith(self.name):
                download_google_url(self.dataset_drive_url[fname], self.processed_dir, filename=fname)
        for fname in self.splits_drive_url:
            if fname.startswith(self.name):
                download_google_url(self.splits_drive_url[fname], self.processed_dir, filename=fname)

    def process(self) -> None:
        if self.name in ['twitch-gamer']:
            x = pd.read_csv(osp.join(self.raw_dir, f'{self.name}_features.csv'))
            n = len(x)
            x = x.drop('numeric_id', axis=1)
            x['created_at'] = x.created_at.replace('-', '', regex=True).astype(int)
            x['updated_at'] = x.updated_at.replace('-', '', regex=True).astype(int)
            one_hot = {k: v for v, k in enumerate(x['language'].unique())}
            x['language'] = [one_hot[lang] for lang in x['language']]
            y = torch.tensor(x['dead_account'].to_numpy()).flatten()
            x = torch.tensor(x.drop('dead_account', axis=1).to_numpy(), dtype=torch.float)
            x = (x - x.mean(0)) / x.std(0)
            edge_index = pd.read_csv(osp.join(self.raw_dir, f'{self.name}_edges.csv'))
            edge_index = torch.tensor(edge_index.to_numpy()).t().type(torch.LongTensor).contiguous()
            edge_index = coalesce(edge_index, num_nodes=n)
        elif self.name in ['wiki']:
            x = torch.load(osp.join(self.raw_dir, f'{self.name}_features.pt'))
            y = torch.load(osp.join(self.raw_dir, f'{self.name}_views.pt'))
            n = y.shape[0]
            edge_index = torch.load(osp.join(self.raw_dir, f'{self.name}_edges.pt')).t().contiguous()
            edge_index = coalesce(edge_index, num_nodes=n)
        else:
            data = scipy.io.loadmat(osp.join(self.raw_dir, f'{self.name}.mat'))
            x = torch.tensor(data['node_feat'], dtype=torch.float)
            if self.name in ['snap-patents']:
                y = even_quantile_labels(data['year'].flatten(), NCLASS_Q, verbose=False)
            else:
                y = torch.tensor(data['label'], dtype=torch.long).flatten()
            n = y.shape[0]
            edge_index = torch.tensor(data['edge_index'], dtype=torch.long).contiguous()
            edge_index = coalesce(edge_index, num_nodes=n)
        kwargs = {'x': x, 'edge_index': edge_index, 'y': y}

        splits_keys = [k.split('_')[0] for k in self.splits_drive_url]
        if self.name in splits_keys:
            # 50/25/25 train/valid/test split
            splits_lst = np.load(osp.join(self.raw_dir, f'{self.name}_splits.npy'), allow_pickle=True)
            mask = {}
            for k in ['train', 'valid', 'test']:
                mask[k] = torch.zeros((n, len(splits_lst)), dtype=torch.bool)
                for i in range(len(splits_lst)):
                    mask[k][splits_lst[i][k], i] = True
            kwargs['train_mask'] = mask['train']
            kwargs['val_mask'] = mask['valid']
            kwargs['test_mask'] = mask['test']

        data = Data(**kwargs)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])


class FB100(InMemoryDataset):
    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        self.name = name.lower()
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        return f'{self.name}.mat'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        url = "https://github.com/CUAI/Non-Homophily-Large-Scale/raw/master/data/facebook100"
        name = self.name.replace('_', ' ').capitalize()
        download_url(f'{url}/{name}.mat', self.raw_dir, filename=f'{self.name}.mat')

    def process(self) -> None:
        data = scipy.io.loadmat(osp.join(self.raw_dir, f'{self.name}.mat'))
        n = data['A'].shape[0]
        edge_index = torch.tensor(data['A'].nonzero(), dtype=torch.long).contiguous()
        edge_index = coalesce(edge_index, num_nodes=n)

        data = data['local_info'].astype(int)
        y = torch.tensor(data[:, 1] - 1, dtype=torch.long).flatten()
        # make features into one-hot encodings
        data = np.hstack((np.expand_dims(data[:, 0], 1), data[:, 2:]))
        x = np.empty((n, 0))
        for col in range(data.shape[1]):
            feat_col = data[:, col]
            feat_onehot = label_binarize(feat_col, classes=np.unique(feat_col))
            x = np.hstack((x, feat_onehot))
        x = torch.tensor(x, dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, y=y)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])


class T_arxiv_year(BaseTransform):
    def forward(self, data: Any) -> Any:
        y = even_quantile_labels(data['node_year'].flatten(), NCLASS_Q, verbose=False)
        data.y = torch.tensor(y, dtype=torch.long)
        del data['node_year']
        return data


class T_ogbn_mag(BaseTransform):
    def forward(self, data: Any) -> Any:
        new_data = Data(
            x=data.x_dict['paper'],
            edge_index=data.edge_index_dict[('paper', 'cites', 'paper')],
            y=data.y_dict['paper'],
            num_nodes=data.x_dict['paper'].shape[0])
        return new_data
