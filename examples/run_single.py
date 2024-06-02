# -*- coding:utf-8 -*-
"""Run with a single data+model+conv+hyperparam with list of seeds.
Author: nyLiao
File Created: 2023-08-03
"""
import logging

from trainer import SingleGraphLoader, ModelLoader, SingleFilterLoader
from utils import (
    setup_seed,
    setup_argparse,
    setup_args,
    save_args,
    setup_logger,
    setup_logpath,
    clear_logger,
    ResLogger)


def main(args):
    # ========== Run configuration
    logger = setup_logger(args.logpath, level_console=args.loglevel, quiet=args.quiet)
    res_logger = ResLogger(quiet=args.quiet)
    res_logger.concat([('seed', args.seed),])

    # ========== Run classification
    if args.task == 'classification':
        # ========== Load data
        data_loader = SingleGraphLoader(args, res_logger)
        data, metric = data_loader(args)
    elif args.task == 'filtering':
        # ========== Load data
        data_loader = SingleFilterLoader(args, res_logger)
        data, metric = data_loader(args)

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
    logger.log(logging.LRES, f"[res]: {res_logger}")
    res_logger.save()
    save_args(args.logpath, vars(args))
    clear_logger(logger)


if __name__ == '__main__':
    parser = setup_argparse()
    args = setup_args(parser)

    seed_lst = args.seed.copy()
    for seed in seed_lst:
        args.seed = setup_seed(seed, args.cuda)
        args.flag = f'{args.seed}'
        args.logpath, args.logid = setup_logpath(
            folder_args=(args.data, args.model_repr, args.conv_repr, args.flag),
            quiet=args.quiet)

        main(args)
