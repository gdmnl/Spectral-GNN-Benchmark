# -*- coding:utf-8 -*-
"""Run single experiment.
Author: nyLiao
File Created: 2023-08-03
"""
import logging

from trainer import (
    DatasetLoader,
    ModelLoader,
    TrnFullbatchIter)
from utils import (
    setup_argparse,
    setup_args,
    save_args,
    setup_logger,
    setup_logpath,
    clear_logger,
    ResLogger)


def main(args):
    # ========== Run configuration
    args.logpath = setup_logpath(
        folder_args=(args.data, args.model, args.flag),
        quiet=args.quiet)
    logger = setup_logger(args.logpath, level=args.loglevel, quiet=args.quiet)
    res_logger = ResLogger(args.logpath.parent.parent, quiet=args.quiet)

    logger.info(f"[args]: {args}")
    res_logger.concat([('seed', args.seed),])
    save_args(args.logpath, args)

    # ========== Load data
    data_loader = DatasetLoader(args, res_logger)
    dataset = data_loader(args)

    # ========== Load model
    model_loader = ModelLoader(args, res_logger)
    model = model_loader(args)

    # ========== Run trainer
    trn = TrnFullbatchIter(
        model=model,
        dataset=dataset,
        args=args,
        res_logger=res_logger,)
    del model, dataset
    trn()

    res_logger.save()
    logger.log(logging.LRES, f"[res]: {res_logger}")
    clear_logger(logger)


if __name__ == '__main__':
    parser = setup_argparse()
    # Experiment-specific arguments
    # parser.add_argument()
    args = setup_args(parser)

    main(args)
