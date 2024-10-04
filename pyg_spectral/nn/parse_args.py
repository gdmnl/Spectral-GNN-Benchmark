from functools import wraps

from .conv.base_mp import BaseMP
from .models.base_nn import BaseNN
from .models_pyg import model_regi_pyg, conv_regi_pyg


def update_regi(regi, new_regi):
    for k in regi:
        regi[k].update(new_regi[k])
    return regi


conv_regi = BaseMP.register_classes()
conv_regi = update_regi(conv_regi, conv_regi_pyg)
model_regi = BaseNN.register_classes()
model_regi = update_regi(model_regi, model_regi_pyg)

full_pargs = set(v for pargs in conv_regi['pargs'].values() for v in pargs)
full_pargs.update(v for pargs in model_regi['pargs'].values() for v in pargs)

compose_name = {
    'ACMGNN': {
        'ACMConv-1-low-high':     'FBGNNI',
        'ACMConv-2-low-high':     'FBGNNII',
        'ACMConv-1-low-high-id':  'ACMGNNI',
        'ACMConv-2-low-high-id':  'ACMGNNII',},
    'DecoupledFixedCompose': {
        'AdjiConv,AdjiConv-ones,ones': 'FAGNN',
        'Adji2Conv,Adji2Conv-gaussian,gaussian': 'G2CN',
        'AdjDiffConv,AdjDiffConv-appr,appr': 'GNN-LFHF',},
    'DecoupledVarCompose': {
        'AdjConv,ChebConv,BernConv': 'FiGURe',},
    'PrecomputedFixedCompose': {
        'AdjSkipConv,AdjSkipConv-ones,ones': 'FAGNN',
        'AdjSkip2Conv,AdjSkip2Conv-gaussian,gaussian': 'G2CN',
        'AdjDiffConv,AdjDiffConv-appr,appr': 'GNN-LFHF',},
    'PrecomputedVarCompose': {
        'AdjConv,ChebConv,BernConv': 'FiGURe',}
}


def resolve_func(nargs=1):
    """Enable calling the return function with additional arguments.

    Args:
        nargs: The number of arguments to pass to the decorated function. Arguments
        beyond this number will be passed to the return function.
    Examples:
        ```python
        @resolve_func(1)
        def foo(bar):
            return bar
        foo(1)  # 1
        foo(lambda x: x+1, 2)  # 3
        ```
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*inputs):
            if len(inputs) <= nargs:
                return func(*inputs[:nargs])
            else:
                ret = func(*inputs[:nargs])
                if callable(ret):
                    return ret(*inputs[nargs:])
                return ret
        return wrapper
    return decorator


@resolve_func(2)
def get_dct(dct: dict, k: str) -> str:
    return dct[k]


def get_model_regi(model: str, k: str, args=None) -> str:
    r"""Getter for :attr:`model_regi`.

    Args:
        model: The name of the model.
        k: The key of the model registry.
        args: Configuration arguments.
    Returns:
        value (str): The value of the model registry.
    """
    return get_dct(model_regi[k], model, args)


def get_conv_regi(conv: str, k: str, args=None) -> str:
    r"""Getter for :attr:`conv_regi`.

    Args:
        conv: The name of the convolution.
        k: The key of the convolution registry.
        args: Configuration arguments.
    Returns:
        value (str): The value of the convolution registry.
    """
    return get_dct(conv_regi[k], conv, args)


def get_nn_name(model: str, conv: str, args) -> str:
    r"""Parse model+conv name for logging path from argparse input.

    Args:
        model: Input argparse string of model.
        conv: Input argparse string of conv. Can be composed.
        args: Additional arguments specified in module :attr:`name` functions.
    Returns:
        nn_name (Tuple[str]): Name strings ``(model_name, conv_name)``.
    """
    model_name = get_dct(model_regi['name'], model, args)
    conv_name = [get_dct(conv_regi['name'], channel, args) for channel in conv.split(',')]
    conv_name = ','.join(conv_name)
    conv_name = get_dct(model_regi['conv_name'], model, conv_name, args)
    if model in compose_name:
        if conv_name in compose_name[model]:
            conv_name = compose_name[model][conv_name]
    return (model_name, conv_name)


def set_pargs(model: str, conv: str, args):
    r"""Build required init arguments (1) from argparse input, then (2) from
    default dict. Delete unused arguments from :attr:`args`.

    Args:
        model: The name of the model.
        conv: The type of convolution. Can be composed.
        args: Configuration arguments.
    Returns:
        kwargs (dict): Arguments for importing the model.
    """
    valid_pargs = model_regi['pargs'][model]
    valid_pargs.extend(conv_regi['pargs'][channel] for channel in conv.split(','))

    kwargs = {}
    for parg in full_pargs:
        if parg in valid_pargs and hasattr(args, parg):
            kwargs[parg] = getattr(args, parg)
        else:
            delattr(args, parg)

    if model in model_regi['pargs_default']:
        for parg in model_regi['pargs_default'][model]:
            kwargs.setdefault(parg, get_dct(model_regi['pargs_default'][model], parg, kwargs))
    for channel in conv.split(','):
        if channel in conv_regi['pargs_default']:
            for parg in conv_regi['pargs_default'][channel]:
                kwargs.setdefault(parg, get_dct(conv_regi['pargs_default'][channel], parg, kwargs))
    return kwargs


def get_param(model: str, conv: str, parg: str, args) -> tuple:
    r"""Query parameter settings for model+conv.

    Args:
        model: The name of the model.
        conv: The type of convolution.
        parg: The name key of the parameter.
        args: Configuration arguments.
    Returns:
        tune_tuple (dict): Configurations for tuning the model.
    """
    if parg in model_regi['param'][model]:
        return get_dct(model_regi['param'][model], parg, args)
    return get_dct(conv_regi['param'][conv], parg, args)
