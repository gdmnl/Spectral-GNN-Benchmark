from torch_geometric.data import InMemoryDataset
import torch
from torch_geometric.data.data import Data
import scipy.io as sio
import numpy as np
import os.path as osp


class Grid2D(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name.lower()
        super(Grid2D, self).__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return ["2Dgrid.mat"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        b=self.processed_paths[0]
        a=sio.loadmat(self.raw_paths[0]) #'subgraphcount/randomgraph.mat')
        # list of adjacency matrix
        A=a['A']
        # list of output
        F=a['F']
        F=F.astype(np.float32)
        #Y=a['Y']
        #Y=Y.astype(np.float32)
        M=a['mask']
        M=M.astype(np.float32)

        data_list = []
        E=np.where(A>0)
        edge_index=torch.Tensor(np.vstack((E[0],E[1]))).type(torch.int64)
        x=torch.tensor(F)
        #y=torch.tensor(Y)
        m=torch.tensor(M)

        train_masks = torch.tensor(M, dtype=torch.bool)
        val_masks = torch.tensor(1-M, dtype=torch.bool)
        test_masks = torch.tensor(1-M, dtype=torch.bool)

        # x_tmp=x[:,0:1]
        # x = x[:, 0:1]

        data_list.append(Data(edge_index=edge_index, x=torch.tensor(F)[:,0:1], m=m, train_mask=train_masks,
                    val_mask=val_masks, test_mask=test_masks))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
