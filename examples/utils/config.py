# -*- coding:utf-8 -*-
"""
Author: nyLiao
File Created: 2023-03-20
File: config.py
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
    parser = argparse.ArgumentParser(description='Benchmark running')
    # Logging configuration
    parser.add_argument('-s', '--seed', type=int, default=0, help='random seed')
    parser.add_argument('-v', '--dev', type=int, default=0, help='GPU id')
    parser.add_argument('-z', '--suffix', type=str, default=None, help='Save name suffix.')
    parser.add_argument('-q', '--quiet', type=bool, default=False, help='Quiet run without saving logs.')
    # Data configuration
    parser.add_argument('-d', '--data', type=str, default='cora', help='Dataset name')
    # Model configuration
    parser.add_argument('-m', '--model', type=str, default='IterConv', help='Model class name')
    parser.add_argument('-c', '--conv', type=str, default='FixLinSumAdj', help='Conv class name')
    parser.add_argument('-l', '--layer', type=int, default=2, help='Number of layers')
    parser.add_argument('-w', '--hidden', type=int, default=256, help='Number of hidden units')
    parser.add_argument('--dp', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--dpe', type=float, default=0.0, help='Edge dropout rate')
    # Training configuration
    parser.add_argument('-e', '--epoch', type=int, default=20, help='Number of epochs')
    parser.add_argument('-p', '--patience', type=int, default=50, help='Patience epoch for early stopping')
    parser.add_argument('--period', type=int, default=-1, help='Periodic saving epoch interval')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--wd', type=float, default=1e-5, help='Weight decay')

    return parser


def setup_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    # Check args
    args = parser.parse_args()
    args = setup_cuda(args)
    args.seed = setup_seed(args.seed, args.cuda)
    # Set new args
    args.flag = f'{args.seed}'
    return args


def save_args(logpath: Path, args: argparse.Namespace):
    if args.quiet:
        return
    with open(logpath.joinpath('config.json'), 'w') as f:
        f.write(json.dumps(dict_to_json(vars(args)), indent=4))


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
