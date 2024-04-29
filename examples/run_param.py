# -*- coding:utf-8 -*-
"""Run with a single data+model+conv entry using optuna for tuning hyperparameters.
Author: nyLiao
File Created: 2024-04-29
"""
import logging
from copy import deepcopy
import json
import optuna

from trainer import SingleGraphLoader, ModelLoader
from utils import (
    force_list_str,
    setup_seed,
    setup_argparse,
    setup_args,
    save_args,
    setup_logger,
    setup_logpath,
    clear_logger,
    ResLogger,
    LOGPATH)

METRIC_NAME = 'f1micro_val'


def _get_suggest(trial, key):
    suggest_dct = {
        'normg': trial.suggest_float('normg', 0.0, 1.0),
        'dp': trial.suggest_float('dp', 0.0, 1.0),
        'lr': trial.suggest_float('lr', 1e-5, 1e-1, log=True),
        'wd': trial.suggest_float('wd', 1e-6, 1e-3, log=True),
    }
    return suggest_dct[key]


def run_single(args):
    # ========== Run configuration
    logger = logging.getLogger('log')
    res_logger = ResLogger(args.logpath, prefix='param', quiet=args.quiet)
    res_logger.concat([('seed', args.seed),])
    for key in args.param:
        res_logger.concat([(key, args.__dict__[key], type(args.__dict__[key]))])

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
    logger.info(f"[res]: {res_logger}\n")

    # ========== Find best
    return res_logger.data.loc[0, METRIC_NAME]


def objective(args, trial):
    args_single = deepcopy(args)
    args.quiet = True
    for key in args.param:
        args_single.__dict__[key] = _get_suggest(trial, key)
    return run_single(args_single)


def main(args):
    args.flag = f'param-{args.seed}'
    args.logpath = setup_logpath(
        folder_args=(args.model, args.data, args.conv, args.flag),
        quiet=args.quiet)
    logger = setup_logger(args.logpath, level_console=args.loglevel, level_file=20, quiet=args.quiet)

    storage_path = LOGPATH.joinpath('optuna.db').resolve().absolute()
    study = optuna.create_study(
        study_name='-'.join([args.model, args.data, args.conv, args.flag]),
        storage=f'sqlite:///{str(storage_path)}',
        direction='maximize',
        load_if_exists=True)
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    study.optimize(
        lambda trial: objective(args, trial),
        n_trials=args.n_trials,
        gc_after_trial=True,
        show_progress_bar=True,)
    clear_logger(logger)


if __name__ == '__main__':
    parser = setup_argparse()
    # Experiment-specific arguments
    parser.add_argument('--param', type=force_list_str, help='List of hyperparameters to search')
    parser.add_argument('--n_trials', type=int, default=5, help='Number of trials')
    args = setup_args(parser)

    seed_lst = args.seed.copy()
    for seed in seed_lst:
        args.seed = setup_seed(seed, args.cuda)

        main(args)
