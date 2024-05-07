# -*- coding:utf-8 -*-
"""Run with a single data+model+conv loading hyperparams from optuna.
Author: nyLiao
File Created: 2024-04-29
"""
import logging
import optuna

from trainer import SingleGraphLoader, ModelLoader
from utils import (
    setup_seed,
    setup_argparse,
    setup_args,
    save_args,
    setup_logger,
    setup_logpath,
    clear_logger,
    ResLogger,
    LOGPATH)


def main(args):
    # ========== Run configuration
    args.logpath = setup_logpath(
        folder_args=(args.model, args.data, args.conv_str, args.flag),
        quiet=args.quiet)
    logger = setup_logger(args.logpath, level_console=args.loglevel, quiet=args.quiet)
    res_logger = ResLogger(args.logpath.parent.parent, quiet=args.quiet)
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
    logger.log(logging.LRES, f"[res]: {res_logger}")
    res_logger.save()
    save_args(args.logpath, args)
    clear_logger(logger)


if __name__ == '__main__':
    parser = setup_argparse()
    # Experiment-specific arguments
    parser.add_argument('--seed_param', type=int, default=1, help='Seed for optuna search')
    args = setup_args(parser)

    storage_path = '-'.join(filter(None, ['optuna', args.suffix])) + '.db'
    storage_path = LOGPATH.joinpath(storage_path).resolve().absolute()
    study = optuna.load_study(
        study_name='-'.join([args.model, args.data, args.conv_str, f'param-{args.seed_param}']),
        storage=f'sqlite:///{str(storage_path)}')
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    best_params = study.best_params
    for key, value in best_params.items():
        setattr(args, key, value)

    seed_lst = args.seed.copy()
    for seed in seed_lst:
        args.seed = setup_seed(seed, args.cuda)
        args.flag = f'{args.seed}'

        main(args)
