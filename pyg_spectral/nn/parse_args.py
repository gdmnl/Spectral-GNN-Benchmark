from .conv.base_mp import BaseMP
from .models.base_nn import BaseNN
from .models_pyg import model_regi_pyg, conv_regi_pyg
from pyg_spectral.utils import CallableDict


def update_regi(regi, new_regi):
    for k in regi:
        regi[k].update(new_regi[k])
    return regi


conv_regi = BaseMP.register_classes()
conv_regi = update_regi(conv_regi, conv_regi_pyg)
model_regi = BaseNN.register_classes()
model_regi = update_regi(model_regi, model_regi_pyg)

conv_regi  = CallableDict.to_callableVal(conv_regi, reckeys=['pargs_default', 'param'])
r'''Fields:
    * name (CallableDict[str, str]): Conv class logging path name.
    * pargs (CallableDict[str, list[str]]): Conv arguments from argparse.
    * pargs_default (dict[str, CallableDict[str, Any]]): Default values for model arguments. Not recommended.
    * param (dict[str, CallableDict[str, ParamTuple]]): Conv parameters to tune.
'''
model_regi = CallableDict.to_callableVal(model_regi, reckeys=['pargs_default', 'param'])
r'''Fields:
    name (CallableDict[str, str]): Model class logging path name.
    conv_name (CallableDict[str, Callable[[str, Any], str]]): Wrap conv logging path name.
    module (CallableDict[str, str]): Module for importing the model.
    pargs (CallableDict[str, list[str]]): Model arguments from argparse.
    pargs_default (dict[str, CallableDict[str, Any]]): Default values for model arguments. Not recommended.
    param (dict[str, CallableDict[str, ParamTuple]]): Model parameters to tune.
'''

full_pargs = set(v for pargs in conv_regi['pargs'].values() for v in pargs)
full_pargs.update(v for pargs in model_regi['pargs'].values() for v in pargs)

compose_name = {
    'ACMGNN': {
        'ACMConv-1-low-high':     'FBGNNI',
        'ACMConv-2-low-high':     'FBGNNII',
        'ACMConv-1-low-high-id':  'ACMGNNI',
        'ACMConv-2-low-high-id':  'ACMGNNII',},
    'ACMGNNDec': {
        'ACMConv-1-low-high':     'FBGNNI',
        'ACMConv-2-low-high':     'FBGNNII',
        'ACMConv-1-low-high-id':  'ACMGNNI',
        'ACMConv-2-low-high-id':  'ACMGNNII',},
    'DecoupledFixedCompose': {
        'AdjiConv,AdjiConv-ones,ones': 'FAGNN',
        'Adji2Conv,Adji2Conv-gaussian,gaussian': 'G2CN',
        'AdjDiffConv,AdjDiffConv-appr,appr': 'GNN_LFHF',},
    'DecoupledVarCompose': {
        'AdjConv,ChebConv,BernConv': 'FiGURe',},
    'PrecomputedFixedCompose': {
        'AdjSkipConv,AdjSkipConv-ones,ones': 'FAGNN',
        'AdjSkip2Conv,AdjSkip2Conv-gaussian,gaussian': 'G2CN',
        'AdjDiffConv,AdjDiffConv-appr,appr': 'GNN_LFHF',},
    'PrecomputedVarCompose': {
        'AdjConv,ChebConv,BernConv': 'FiGURe',}
}
compose_param = {
    'DecoupledFixedCompose': {
        'G2CN': CallableDict({
            'beta':  [('float', (1.00, 2.00), {'step': 0.01}, lambda x: round(x, 2)),
                      ('float', (0.01, 1.00), {'step': 0.01}, lambda x: round(x, 2))],}),
        'GNN_LFHF': CallableDict({
            'beta':  [('float', ( 0.01,  1.00), {'step': 0.01}, lambda x: round(x, 2)),
                      ('float', (-1.00, -0.01), {'step': 0.01}, lambda x: round(x, 2))],}),
    },
    'PrecomputedFixedCompose': {
        'G2CN': CallableDict({
            'beta':  [('float', (1.00, 2.00), {'step': 0.01}, lambda x: round(x, 2)),
                      ('float', (0.01, 1.00), {'step': 0.01}, lambda x: round(x, 2))],}),
        'GNN_LFHF': CallableDict({
            'beta':  [('float', ( 0.01,  1.00), {'step': 0.01}, lambda x: round(x, 2)),
                      ('float', (-1.00, -0.01), {'step': 0.01}, lambda x: round(x, 2))],}),
    },
}


def get_model_regi(model: str, k: str, args=None) -> str:
    r"""Getter for :attr:`model_regi`.

    Args:
        model: The name of the model.
        k: The key of the model registry.
        args: Configuration arguments.
    Returns:
        value (str): The value of the model registry.
    """
    if not model in model_regi[k]:
        return None
    return model_regi[k](model, args) if args else model_regi[k][model]


def get_conv_regi(conv: str, k: str, args=None) -> str:
    r"""Getter for :attr:`conv_regi`.

    Args:
        conv: The name of the convolution.
        k: The key of the convolution registry.
        args: Configuration arguments.
    Returns:
        value (str): The value of the convolution registry.
    """
    if not conv in conv_regi[k]:
        return None
    return conv_regi[k](conv, args) if args else conv_regi[k][conv]


def get_conv_subregi(conv: str, k: str, pargs: str, args=None) -> str:
    r"""Getter for calling a sub-CallableDict in :attr:`conv_regi`.

    Args:
        conv: The name of the convolution.
        k: The key in :attr:`conv_regi`.
        pargs: The key in the sub-CallableDict.
        args: Configuration arguments.
    Returns:
        value (str): The value of the sub-CallableDict.
    """
    if ',' in conv:
        return [conv_regi[k][channel](pargs, args) for channel in conv.split(',')]
    return conv_regi[k][conv](pargs, args) if args else conv_regi[k][conv][pargs]


def get_nn_name(model: str, conv: str, args) -> str:
    r"""Parse model+conv name for logging path from argparse input.

    Args:
        model: Input argparse string of model.
        conv: Input argparse string of conv. Can be composed.
        args: Additional arguments specified in module :attr:`name` functions.
    Returns:
        nn_name (tuple[str]): Name strings ``(model_name, conv_name)``.
    """
    model_name = model_regi['name'](model, args)
    conv_name = [conv_regi['name'](channel, args) for channel in conv.split(',')]
    conv_name = ','.join(conv_name)
    conv_name = model_regi['conv_name'](model, conv_name, args)
    if model in compose_name and conv_name in compose_name[model]:
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
    for channel in conv.split(','):
        valid_pargs.extend(conv_regi['pargs'][channel])

    kwargs = {}
    for parg in full_pargs:
        if hasattr(args, parg):
            if parg in valid_pargs:
                kwargs[parg] = getattr(args, parg)
            else:
                delattr(args, parg)

    if model in model_regi['pargs_default']:
        for parg in model_regi['pargs_default'][model]:
            kwargs.setdefault(parg, model_regi['pargs_default'][model](parg, kwargs))
    for channel in conv.split(','):
        if channel in conv_regi['pargs_default']:
            for parg in conv_regi['pargs_default'][channel]:
                kwargs.setdefault(parg, conv_regi['pargs_default'][channel](parg, kwargs))
    return kwargs
