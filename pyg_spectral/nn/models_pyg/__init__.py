from .iterative import ChebNet, kwvars_ChebNet

kwvars = {
    'ChebNet': kwvars_ChebNet,
}

kwargs_default = {
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
}
