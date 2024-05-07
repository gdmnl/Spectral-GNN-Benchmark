# -*- coding:utf-8 -*-
""" Evaluate a single saved model.
Author: nyLiao
File Created: 2024-05-07
"""
import logging

from trainer import SingleGraphLoader, ModelLoader
from utils import (
    setup_seed,
    setup_argparse,
    setup_args,
    setup_logger,
    setup_logpath,
    clear_logger,
    ResLogger)


def main(args):
    # ========== Run configuration
    logger = setup_logger(args.logpath, level_console=args.loglevel, quiet=True)
    res_logger = ResLogger(quiet=True)
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
    trn._fetch_data()
    trn.model = trn.ckpt_logger.load('best', model=trn.model)
    trn.model = trn.model.to(trn.device)
    res_test = trn.test()

    # logger.info(f"[args]: {args}")
    # logger.log(logging.LRES, f"[res]: {res_logger}")
    clear_logger(logger)


if __name__ == '__main__':
    parser = setup_argparse()
    args = setup_args(parser)

    seed_lst = args.seed.copy()
    for seed in seed_lst:
        args.seed = setup_seed(seed, args.cuda)
        args.flag = f'{args.seed}'
        args.logpath, args.logid = setup_logpath(
            folder_args=(args.data, args.model_repr, args.conv_repr, args.flag),)

        main(args)
