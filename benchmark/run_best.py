# -*- coding:utf-8 -*-
"""Run with a single data+model+conv loading hyperparams from optuna.
Author: nyLiao
File Created: 2024-04-29
"""
import logging
import json

from trainer import SingleGraphLoader, ModelLoader
from utils import (
    setup_seed,
    setup_argparse,
    setup_args,
    save_args,
    setup_logger,
    setup_logpath,
    clear_logger,
    ResLogger)


def reverse_parse(parser, key, val):
    for action in parser._actions:
        if action.dest == key:
            type_func = action.type
            break
    return type_func(val)


def filter_res(s, metric):
    # remove all substring start with s but not contain metric
    flt_common = lambda x: not x.startswith('s_') and not '_' in x
    flt_metric = lambda x: metric in x and x.endswith('_test')
    lst = [x for x in s.split(', ') if flt_common(x.split(':')[0]) or flt_metric(x.split(':')[0])]
    return ', '.join(lst)


def main(args):
    # ========== Run configuration
    logger = setup_logger(args.logpath, level_console=args.loglevel, quiet=args.quiet)
    res_logger = ResLogger(prefix=args.eval_name, quiet=False)
    res_logger.concat([('seed', args.seed),])

    # ========== Load data
    data_loader = SingleGraphLoader(args, res_logger)
    data = data_loader(args)

    # ========== Load model
    model_loader = ModelLoader(args, res_logger)
    model, trn = model_loader(args)
    res_logger.suffix = trn.name

    # ========== Run trainer
    trn = trn(
        model=model,
        data=data,
        args=args,
        res_logger=res_logger,)
    del model, data
    trn()

    logger.info(f"[args]: {args}")
    resstr = filter_res(res_logger.get_str(), args.metric)
    logger.log(logging.LRES, f"{resstr}")
    res_logger.save()
    save_args(args.logpath, vars(args))
    clear_logger(logger)


if __name__ == '__main__':
    parser = setup_argparse()
    # Experiment-specific arguments
    parser.add_argument('--eval_name', type=str, default='summary', help='Exp name')
    parser.add_argument('--seed_param', type=int, default=1, help='Seed for optuna search')
    args = setup_args(parser)

    study_path, _ = setup_logpath(
        folder_args=(args.data, args.model_repr, args.conv_repr, f'param-{args.seed_param}',
                     'config.json'))
    with open(study_path, 'r') as config_file:
        best_params = json.load(config_file)
    for key, value in best_params.items():
        setattr(args, key, reverse_parse(parser, key, value))

    seed_lst = args.seed.copy()
    for seed in seed_lst:
        args.seed = setup_seed(seed, args.cuda)
        args.flag = f'{args.seed}'
        args.logpath, _ = setup_logpath(
            folder_args=(args.data, args.model_repr, args.conv_repr, args.flag),
            quiet=args.quiet)

        main(args)
