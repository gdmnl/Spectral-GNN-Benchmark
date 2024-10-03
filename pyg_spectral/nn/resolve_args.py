from functools import wraps

from .conv import BaseMP
from .models import BaseNN

conv_name_map, conv_pargs, conv_param = BaseMP.register_classes()
model_name_map, wrap_conv_name_map, model_pargs, model_param = BaseNN.register_classes()

full_pargs = set(v for pargs in conv_pargs.values() for v in pargs)
full_pargs.update(v for pargs in model_pargs.values() for v in pargs)

compose_name_map = {
    'ACMGNN': {
        'ACMConv-1.0-low-high':     'FBGNNI',
        'ACMConv-2.0-low-high':     'FBGNNII',
        'ACMConv-1.0-low-high-id':  'ACMGNNI',
        'ACMConv-2.0-low-high-id':  'ACMGNNII',},
    'DecoupledFixedCompose': {
        'AdjiConv,AdjiConv-ones,ones': 'FAGNN',
        'Adji2Conv,Adji2Conv-gaussian,gaussian': 'G2CN',
        'AdjDiffConv,AdjDiffConv-appr,appr': 'GNN-LF/HF',},
    'DecoupledVarCompose': {
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


def get_nn_name(model: str, conv: str, args) -> str:
    r"""Resolves model+conv name for logging path from argparse input.

    Args:
        model: Input argparse string of model.
        conv: Input argparse string of conv.
        args: Additional arguments specified in module :attr:`name` functions.
    Returns:
        nn_name (Tuple[str]): Name strings ``(model_name, conv_name)``.
    """
    model_name = get_dct(model_name_map, model, args)
    conv_name  = get_dct(conv_name_map, conv, args)
    conv_name  = get_dct(wrap_conv_name_map, model, conv_name, args)
    if model in compose_name_map:
        if conv_name in compose_name_map[model]:
            conv_name = compose_name_map[model][conv_name]
    return (model_name, conv_name)


def set_pargs(model: str, conv: str, args):
    r"""Build required init arguments from argparse input. Delete unused arguments.

    Args:
        model: The name of the model.
        conv: The type of convolution.
        args: Configuration arguments.
    Returns:
        kwargs (dict): Arguments for importing the model.
    """
    valid_pargs = model_pargs[model] + conv_pargs[conv]
    kwargs = {}
    for parg in full_pargs:
        if parg in valid_pargs:
            kwargs[parg] = getattr(args, parg)
        else:
            delattr(args, parg)
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
    if parg in model_param[model]:
        return get_dct(model_param[model], parg, args)
    return get_dct(conv_param[conv], parg, args)
