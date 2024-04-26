# -*- coding:utf-8 -*-
"""Run experiments with a single data+model+conv entry with lists of hyperparameters.
Author: nyLiao
File Created: 2024-04-26
"""
import logging
from copy import deepcopy
from pathlib import Path
import json

from trainer import DatasetLoader, ModelLoader
from utils import (
    force_list_str,
    setup_seed,
    setup_argparse,
    setup_args,
    save_args,
    dict_to_json,
    setup_logger,
    setup_logpath,
    clear_logger,
    ResLogger)


def main(args):
    # ========== Configuration
    args.logpath = setup_logpath(
        folder_args=(args.model, args.data, args.conv, args.flag),
        quiet=args.quiet)
    logger = setup_logger(args.logpath, level_console=args.loglevel, level_file=20, quiet=args.quiet)
    res_logger = ResLogger(args.logpath, prefix='param', quiet=args.quiet)
    res_logger.concat([('seed', args.seed),])
    logger.log(logging.LRES, f"[param]: {args}")
    save_args(args.logpath, args)

    # ========== Search
    res_summary = ResLogger(args.logpath, quiet=True)
    recursive_param(deepcopy(args), args.param.copy(), logger, res_logger, res_summary)

    # ========== Find best
    metric = 'f1micro_val'
    row_best = res_summary.data[metric].idxmax()
    logger.log(logging.LRES, f"[best]: {metric}={res_summary._get(col=metric, row=row_best)}")

    args_best = deepcopy(args)
    params_str = "PARAM=(\n"
    for key in args.param:
        value = res_summary._get(col=key, row=row_best)
        args_best.__dict__[key] = value
        params_str  += f'    "--{key}" "{value}"\n'
    params_str += ")\n"
    logger.log(logging.LRES, f"{params_str}")
    with open(Path(args.logpath.parent, 'param.sh'), 'w') as f:
        f.write(params_str)
    with open(args.logpath.joinpath('config_best.json'), 'w') as f:
        f.write(json.dumps(dict_to_json(vars(args_best)), indent=4))

    clear_logger(logger)


def recursive_param(args, param, logger, res_logger, res_summary):
    if len(param) == 0:
        # ========== Load data
        data_loader = DatasetLoader(args, res_logger)
        dataset = data_loader(args)

        # ========== Load model
        model_loader = ModelLoader(args, res_logger)
        model, trn = model_loader(args)

        # ========== Run trainer
        trn = trn(
            model=model,
            dataset=dataset,
            args=args,
            res_logger=res_logger,)
        del model, dataset
        trn()

        logger.log(logging.LRES, f"[res]: {res_logger}")
        logger.info(f"[args]: {args}\n")
        res_logger.save()
        res_summary.merge(res_logger, rows=[res_summary.nrows+1])
        return res_logger

    else:
        key = param.pop()
        for value in args.__dict__[key]:
            args_curret = deepcopy(args)
            args_curret.__dict__[key] = value
            res_current = deepcopy(res_logger)
            res_current.concat([(key, value, type(value))])

            recursive_param(args_curret, param.copy(), logger, res_current, res_summary)


if __name__ == '__main__':
    parser = setup_argparse()
    # Experiment-specific arguments
    parser.add_argument('--param', type=force_list_str, help='List of hyperparameters to search')
    args = setup_args(parser)

    seed_lst = args.seed.copy()
    for seed in seed_lst:
        args.seed = setup_seed(seed, args.cuda)
        args.flag = f'param-{args.seed}'

        main(args)
