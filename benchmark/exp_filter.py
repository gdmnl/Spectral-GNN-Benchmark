# -*- coding:utf-8 -*-
import logging

from trainer.filter import FilterLoader, TrnFilter
from trainer import ModelLoader
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

    # ========== Load data
    data_loader = FilterLoader(args, res_logger)
    data, metric = data_loader(args)

    # ========== Load model
    model_loader = ModelLoader(args, res_logger)
    model, _ = model_loader(args)
    trn = TrnFilter
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
    parser.add_argument('--filter_type', type=str, choices=['low','high','band','rejection','comb','low_band'], default='band')
    parser.add_argument('--img_idx', type=int, default=0, help='filtering image index')
    args = setup_args(parser)

    seed_lst = args.seed.copy()
    for seed in seed_lst:
        args.seed = setup_seed(seed, args.cuda)
        args.flag = f'{args.seed}'
        args.logpath, args.logid = setup_logpath(
            folder_args=(args.data, args.model_repr, args.conv_repr, args.flag),
            quiet=args.quiet)

        main(args)
