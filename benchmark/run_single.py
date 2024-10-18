# -*- coding:utf-8 -*-
"""Run with a single data+model+conv+hyperparam with list of seeds.
Author: nyLiao
File Created: 2023-08-03
"""
import logging

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
    ResLogger)


def reverse_parse(parser, key, val):
    r"""Set value to the type specified in parser.

    Args:
        parser (argparse.ArgumentParser): The argument parser.
        key (str): The key of value in parser.
        val (str): The value to be converted.
    Returns:
        The value converted to the type specified by the parser for the given key.
    """
    for action in parser._actions:
        if action.dest == key:
            type_func = action.type
            break
    return type_func(val)


def main(args):
    # ========== Run configuration
    logger = setup_logger(args.logpath, level_console=args.loglevel, quiet=args.quiet)
    res_logger = ResLogger(suffix=args.suffix, quiet=(args.suffix is None))
    res_logger.concat([('seed', args.seed),])
    if args.param is not None and args.param != ['']:
        res_logger.concat([(key, getattr(args, key)) for key in args.param])

    # ========== Load data
    data_loader = SingleGraphLoader(args, res_logger)
    data = data_loader(args)

    # ========== Load model
    model_loader = ModelLoader(args, res_logger)
    model, trn = model_loader(args)

    # ========== Run trainer
    trn = trn(
        model=model,
        data=data,
        args=args,
        res_logger=res_logger,)
    del model, data
    trn()

    logger.info(f"[args]: {args}")
    logger.log(logging.LRES, f"[res]: {res_logger.flt_str(args.metric)}")
    res_logger.save()
    save_args(args.logpath, vars(args))
    clear_logger(logger)


if __name__ == '__main__':
    parser = setup_argparse()
    parser.add_argument('--param', type=force_list_str, nargs='?', default=None, const='', help='List of hyperparameters to change')
    args = setup_args(parser)

    # If args.param is None: use hyperparameters in args
    if args.param is not None:
        import json
        study_path = setup_logpath(
            folder_args=(args.data, *ModelLoader.get_name(args), 'param',
                        'config.json'))
        with open(study_path, 'r') as config_file:
            best_params = json.load(config_file)

        if args.param == ['']:
            # Load best hyperparameters
            for key, value in best_params.items():
                setattr(args, key, reverse_parse(parser, key, value))
        else:
            # Load best hyperparameters exluding args.param list
            for key, value in best_params.items():
                if key not in args.param:
                    setattr(args, key, reverse_parse(parser, key, value))

    seed_lst = args.seed.copy()
    for seed in seed_lst:
        args.seed = setup_seed(seed, args.cuda)
        args.flag = f'{args.seed}'
        args.logpath = setup_logpath(
            folder_args=(args.data, *ModelLoader.get_name(args), args.flag),
            quiet=args.quiet)

        main(args)
