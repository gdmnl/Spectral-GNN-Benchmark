import numpy as np
import torch
from torch import Tensor
from torch_geometric.typing import Adj
from pyg_spectral.nn.models.precomputed import PrecomputedFixed
from pyg_spectral.propagations import PyPropComp


class CppPrecFixed(PrecomputedFixed):
    r"""Decoupled structure with C++ propagation precomputation.
    Fixed scalar propagation parameters and accumulating precompute results.
    """

    def preprocess(self,
        x: Tensor,
        edge_index: Adj
    ) -> Adj:
        indices = edge_index.col_indices().cpu().detach().numpy()
        indptr = edge_index.crow_indices().cpu().detach().numpy()
        indices = np.array(indices, dtype=np.uint32)
        indptr = np.array(indptr, dtype=np.uint32)

        indices_re = []
        for i in range(1,indptr.shape[0]):
            indices_re += sorted(indices[indptr[i-1]:indptr[i]],key=lambda x:indptr[x+1]-indptr[x])
        indices = np.asarray(indices_re,dtype=np.uint32)

        n = x.size(0)
        seed = 0
        self.py_propcomp = PyPropComp()
        self.py_propcomp.preprocess(indices, indptr, n, seed)

    def convolute(self,
        x: Tensor,
        edge_index: Adj,
    ) -> Tensor:
        pass
        chn = dict(type=0, hop=self.num_hops, dim=self.in_channels, delta=1e-5,
                   alpha=0, rra=0.5, rrb=0.5)
        x = x.cpu().detach().numpy()
        x = x.transpose().astype(np.float32, order='C')
        time_pre = self.py_propcomp.compute(1, [chn], x)
        return torch.from_numpy(x.transpose())
