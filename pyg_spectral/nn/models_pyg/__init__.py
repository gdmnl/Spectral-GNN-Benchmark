from .iterative import ChebNet

from .parse_args import model_regi_pyg, conv_regi_pyg, register_class


# >>>>>>>>>>
for model in [ChebNet, ]:
# <<<<<<<<<<
    register_class(model, model_regi_pyg)
    register_class(model, conv_regi_pyg)
    model_regi_pyg['module'][model.__name__] = '.'.join(model.__module__.split('.')[:-1])
    for k, dv in {'name': lambda _: '', 'pargs': [], 'pargs_default': {}, 'param': {}}.items():
        conv_regi_pyg[k].setdefault(model.__name__, dv)
