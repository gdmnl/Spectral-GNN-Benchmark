# -*- coding:utf-8 -*-
""" Configuration for PyG models.
"""
from pyg_spectral.nn.conv.base_mp import CONV_REGI_INIT
from pyg_spectral.nn.models.base_nn import MODEL_REGI_INIT


def register_class(cls, registry):
    for k in registry:
        if hasattr(cls, k):
            registry[k][cls.__name__] = getattr(cls, k)
    return registry


# Model level
model_regi_pyg = MODEL_REGI_INIT
model_regi_pyg['name'] = {
    # `torch_geometric.nn.models.BasicGNN`
    'GCN':          'IterativeFixed',
    'GraphSAGE':    'IterativeFixed',
    'GIN':          'IterativeFixed',
    'GAT':          'IterativeFixed',
    'PNA':          'IterativeFixed',
    # Resolve MLP schemes
    'MLP': lambda args: args.conv,
}
model_regi_pyg['module'] = {m: 'torch_geometric.nn.models' for m in model_regi_pyg['name']}

model_regi_pyg['conv_name'] = {
    # `torch_geometric.nn.models.BasicGNN`
    'GCN':          lambda *_: 'GCNConv',
    'GraphSAGE':    lambda *_: 'SAGEConv',
    'GIN':          lambda *_: 'GINConv',
    'GAT':          lambda *_: 'GATConv',
    'PNA':          lambda *_: 'PNAConv',
    'MLP':          lambda *_: 'MLP',
}

basic_gnn_pargs = ['conv', 'num_hops', 'in_channels', 'out_channels', 'hidden_channels', 'dropout_lin',]
model_regi_pyg['pargs'] = {
    'GCN':          basic_gnn_pargs,
    'GraphSAGE':    basic_gnn_pargs,
    'GIN':          basic_gnn_pargs,
    'GAT':          basic_gnn_pargs,
    'PNA':          basic_gnn_pargs,
    'MLP':          basic_gnn_pargs + ['in_layers', 'out_layers',],
}

basic_gnn_param = {
    'num_hops':     ('int', (2, 8), {'step': 1}, lambda x: x),
    'hidden_channels':  ('categorical', ([16, 32, 64, 128, 256],), {}, lambda x: x),
    'dropout_lin':  ('float', (0.0, 1.0), {'step': 0.1}, lambda x: round(x, 2)),
}
model_regi_pyg['param'] = {
    'GCN':          basic_gnn_param,
    'GraphSAGE':    basic_gnn_param,
    'GIN':          basic_gnn_param,
    'GAT':          basic_gnn_param,
    'PNA':          basic_gnn_param,
    'MLP':          basic_gnn_param | {
        'in_layers':    ('int', (1, 3), {'step': 1}, lambda x: x),
        'out_layers':   ('int', (1, 3), {'step': 1}, lambda x: x),
    },
}

# Conv level
conv_regi_pyg = CONV_REGI_INIT

conv_pargs_default = {
    'GCN': {
        'cached': False,
        'add_self_loops': False,
        'improved': False,
        'normalize': True,
    },
    'GraphSAGE': {
        'normalize': True,
    },
    'GIN': {
        'eps': 0.0,
        'train_eps': False,
    },
    'GAT': {
        'heads': 8,
        'concat': True,
        'share_weights': False,
        'add_self_loops': False,
    },
    'DecoupledFixed': {
        'num_layers': lambda kwargs: kwargs.pop('in_layers') + kwargs.pop('out_layers'),
    },
    'IterativeFixed': {
        'num_layers': lambda kwargs: kwargs.pop('in_layers') + kwargs['num_hops'] + kwargs.pop('out_layers'),
    },
    'PrecomputedFixed': {
        'num_layers': lambda kwargs: kwargs.pop('out_layers'),
    },
}

for mname in list(model_regi_pyg['name'].keys()) + list(conv_pargs_default.keys()):
    conv_regi_pyg['name'][mname] = lambda _: ''
    conv_regi_pyg['pargs'][mname] = []
    conv_regi_pyg['pargs_default'][mname] = conv_pargs_default.get(mname, {})
    conv_regi_pyg['param'][mname] = {}
