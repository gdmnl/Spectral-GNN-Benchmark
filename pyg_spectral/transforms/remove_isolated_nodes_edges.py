import copy
from collections import defaultdict
from typing import Union, Iterable

import torch

from torch_geometric.data import Data, HeteroData
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform


@functional_transform('remove_isolated_nodes_edges')
class RemoveIsolatedNodesEdges(BaseTransform):
    r"""Removes isolated nodes from the graph
    (functional name: :obj:`remove_isolated_nodes_edges`).
    """
    def __init__(self, edge_keys: Iterable) -> None:
        self.edge_keys = edge_keys

    def forward(
        self,
        data: Union[Data, HeteroData],
    ) -> Union[Data, HeteroData]:
        # Gather all nodes that occur in at least one edge (across all types):
        n_ids_dict = defaultdict(list)
        for edge_store in data.edge_stores:
            if 'edge_index' not in edge_store:
                continue

            if edge_store._key is None:
                src = dst = None
            else:
                src, _, dst = edge_store._key

            n_ids_dict[src].append(edge_store.edge_index[0])
            n_ids_dict[dst].append(edge_store.edge_index[1])

        n_id_dict = {k: torch.cat(v).unique() for k, v in n_ids_dict.items()}

        n_map_dict = {}
        for node_store in data.node_stores:
            if node_store._key not in n_id_dict:
                n_id_dict[node_store._key] = torch.empty(0, dtype=torch.long)

            idx = n_id_dict[node_store._key]            # len:n_new -> val:n_old
            assert data.num_nodes is not None
            mapping = idx.new_zeros(data.num_nodes)     # len:n_old -> val:n_new
            mapping[idx] = torch.arange(idx.numel(), device=mapping.device)
            n_map_dict[node_store._key] = mapping

        for edge_store in data.edge_stores:
            if 'edge_index' not in edge_store:
                continue

            if edge_store._key is None:
                src = dst = None
            else:
                src, _, dst = edge_store._key

            # row, col = edge_store.edge_index[0], edge_store.edge_index[1]
            # mask = torch.isin(row, n_id_dict[src]) & torch.isin(col, n_id_dict[dst])
            # row = n_map_dict[src][row[mask]]
            # col = n_map_dict[dst][col[mask]]
            row = n_map_dict[src][edge_store.edge_index[0]]
            col = n_map_dict[dst][edge_store.edge_index[1]]
            edge_store.edge_index = torch.stack([row, col], dim=0)

        for key in self.edge_keys:
            if hasattr(data, key):
                edge_index = getattr(data, key)
                # row, col = edge_index[0], edge_index[1]
                # mask = torch.isin(row, n_id_dict[src]) & torch.isin(col, n_id_dict[dst])
                # row = n_map_dict[src][row[mask]]
                # col = n_map_dict[dst][col[mask]]
                row = n_map_dict[src][edge_index[0]]
                col = n_map_dict[dst][edge_index[1]]
                setattr(data, key, torch.stack([row, col], dim=0))

        old_data = copy.copy(data)
        for out, node_store in zip(data.node_stores, old_data.node_stores):
            for key, value in node_store.items():
                if key == 'num_nodes':
                    out.num_nodes = n_id_dict[node_store._key].numel()
                elif node_store.is_node_attr(key):
                    out[key] = value[n_id_dict[node_store._key]]
        return data
