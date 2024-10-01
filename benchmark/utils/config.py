# -*- coding:utf-8 -*-
"""
Author: nyLiao
File Created: 2023-03-20
"""
import os
import json
import uuid
import random
import argparse
from pathlib import Path

import numpy as np
import torch


# noinspection PyUnresolvedReferences
def setup_seed(seed: int = None, cuda: bool = True) -> int:
    if seed is None:
        seed = int(uuid.uuid4().hex, 16) % 1000000
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)
    return seed


def setup_cuda(args: argparse.Namespace) -> argparse.Namespace:
    args.cuda = args.dev >= 0 and torch.cuda.is_available()
    args.device = torch.device("cuda:{}".format(args.dev) if args.cuda else "cpu")
    if args.cuda:
        torch.cuda.set_device(args.device)
    return args


def setup_argparse():
    np.set_printoptions(linewidth=160, edgeitems=5, threshold=20,
                        formatter=dict(float=lambda x: f"{x: 9.3e}"))
    torch.set_printoptions(linewidth=160, edgeitems=5)

    parser = argparse.ArgumentParser(description='Benchmark running')
    # Logging configuration
    parser.add_argument('-s', '--seed', type=force_list_int, default=[42], help='random seed')
    parser.add_argument('-v', '--dev', type=int, default=0, help='GPU id')
    parser.add_argument('-z', '--suffix', type=str, default=None, help='Result log file name. None:not saving results')
    parser.add_argument('-quiet', action='store_true', help='File log. True:dry run without saving logs')
    parser.add_argument('--loglevel', type=int, default=10, help='Console log. 10:progress, 15:train, 20:info, 25:result')
    parser.add_argument('--storage', type=str, default='state_gpu', choices=['state_file', 'state_ram', 'state_gpu'], help='Storage scheme for saving the checkpoints')
    # Data configuration
    parser.add_argument('-d', '--data', type=str, default='cora', help='Dataset name')
    parser.add_argument('--data_split', type=str, default='Random_60/20/20', help='Dataset split')
    parser.add_argument('--normg', type=float, default=0.5, help='Generalized graph norm')
    parser.add_argument('--normf', type=int, nargs='?', default=0, const=None, help='Embedding norm dimension. 0: feat-wise, 1: node-wise, None: disable')
    parser.add_argument('--multi', action='store_true', help='True for multi-label classification')
    # Model configuration
    parser.add_argument('-m', '--model', type=str, default='DecoupledVar', help='Model class name')
    parser.add_argument('-c', '--conv', type=str, default='AdjConv', help='Conv class name')
    parser.add_argument('-k', '--num_hops', type=int, default=10, help='Number of conv hops')
    parser.add_argument('-l1', '--in_layers',  type=int, default=1, help='Number of MLP layers before conv')
    parser.add_argument('-l2', '--out_layers', type=int, default=1, help='Number of MLP layers after conv')
    parser.add_argument('-w', '--hidden', type=int, default=128, help='Number of hidden width')
    parser.add_argument('--dp_lin', type=float, default=0.5, help='Dropout rate for linear')
    parser.add_argument('--dp_conv', type=float, default=0.5, help='Dropout rate for conv')
    # Training configuration
    parser.add_argument('-e', '--epoch', type=int, default=20, help='Number of epochs')
    parser.add_argument('-p', '--patience', type=int, default=50, help='Patience epoch for early stopping')
    parser.add_argument('--period', type=int, default=-1, help='Periodic saving epoch interval')
    parser.add_argument('-b', '--batch', type=int, default=4096, help='Batch size')
    parser.add_argument('--lr_lin', type=float, default=1.0e-2, help='Learning rate for linear')
    parser.add_argument('-lr', '--lr_conv', type=float, default=1.0e-3, help='Learning rate for conv')
    parser.add_argument('--wd_lin', type=float, default=5e-6, help='Weight decay for linear')
    parser.add_argument('-wd', '--wd_conv', type=float, default=5e-6, help='Weight decay for conv')

    # >>>>>>>>>>
    # Model-specific
    # - Decoupled, ACMGNN
    parser.add_argument('--theta_scheme', type=str, default="ones", help='Filter name')
    parser.add_argument('--theta_param', type=list_float, default=1.0, help='Hyperparameter for filter') # Support list by default
    # - DecoupledCompose
    parser.add_argument('--combine', type=str, default="sum_weighted", choices=['sum', 'sum_weighted', 'cat'], help='How to combine different channels of convs')

    # Conv-specific
    # - AdjConv, ChebConv, Horner, Clenshaw, ACMGNN
    parser.add_argument('--alpha', type=list_float, help='Decay factor')
    # - AdjiConv
    parser.add_argument('--beta', type=list_float, help='Scaling factor')
    # <<<<<<<<<<

    # Test flags
    parser.add_argument('--test_deg', action='store_true', help='Call TrnFullbatch.test_deg()')
    return parser


def setup_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    # Check args
    args = parser.parse_args()
    args = setup_cuda(args)
    # Set new args
    if args.model in ['DecoupledFixed', 'DecoupledFixedCompose']:
        args.model_repr = 'DecoupledFixed'
        args.conv_repr = f'{args.conv}-{args.theta_scheme}'
    elif args.model in ['PrecomputedFixed', 'PrecomputedFixedCompose']:
        args.model_repr = 'PrecomputedFixed'
        args.conv_repr = f'{args.conv}-{args.theta_scheme}'
    elif args.model in ['Iterative', 'IterativeCompose']:
        args.model_repr = 'IterativeFixed'
        args.conv_repr = f'{args.conv}-{args.theta_scheme}'
    elif args.model in ['IterativeCompose']:
        args.model_repr = 'Iterative'
        args.conv_repr = args.conv
    elif args.model in ['DecoupledVar', 'DecoupledVarCompose']:
        args.model_repr = 'DecoupledVar'
        args.conv_repr = args.conv
    elif args.model in ['PrecomputedVar', 'PrecomputedVarCompose']:
        args.model_repr = 'PrecomputedVar'
        args.conv_repr = args.conv
    # FIXME: separate arch for AdaGNN
    elif args.model in ['AdaGNN']:
        args.model_repr = 'DecoupledVar'
        args.conv_repr = args.conv
    elif args.model in ['ACMGNN']:
        args.model_repr = 'Iterative'
        args.conv_repr = f'{args.conv}-{args.alpha}-{args.theta_scheme}'
    elif args.model in ['ACMGNNDec']:
        args.model_repr = 'DecoupledVar'
        args.conv_repr = f'{args.conv}-{args.alpha}-{args.theta_scheme}'
    elif args.model in ['GCN', 'GraphSAGE', 'GIN', 'GAT', 'PNA', 'MLP']:
        args.model_repr = {
            'ones': 'DecoupledFixed',
            'appr': 'IterativeFixed',
            'impulse': 'PrecomputedFixed',
            }[args.theta_scheme]
        args.conv_repr = args.model
    elif args.model in ['ChebNet',]:
        args.model_repr = 'DecoupledFixed'
        args.conv_repr = args.model
    else:
        args.model_repr = args.model
        args.conv_repr = args.conv
    return args


def save_args(logpath: Path, args: dict):
    if 'quiet' in args and args['quiet']:
        return
    with open(logpath.joinpath('config.json'), 'w') as f:
        f.write(json.dumps(dict_to_json(args), indent=4))


def dict_to_json(dictionary) -> dict:
    def is_serializable(obj):
        try:
            json.dumps(obj)
            return True
        except:
            return False

    filtered_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, dict):
            filtered_value = dict_to_json(value)
        elif isinstance(value, list):
            filtered_value = [v for v in value if is_serializable(v)]
        elif is_serializable(value):
            filtered_value = value
        else:
            try:
                filtered_value = str(value)
            except:
                continue
        filtered_dict[key] = filtered_value
    return filtered_dict


force_list_str = lambda x: [str(v) for v in x.split(',')]
force_list_int = lambda x: [int(v) for v in x.split(',')]
list_str = lambda x: [str(v) for v in x.split(',')] if isinstance(x, str) and ',' in x else str(x)
list_int = lambda x: [int(v) for v in x.split(',')] if isinstance(x, str) and ',' in x else int(x)
list_float = lambda x: [float(v) for v in x.split(',')] if isinstance(x, str) and ',' in x else float(x)
