# -*- coding:utf-8 -*-
"""Run single experiment.
Author: nyLiao
File Created: 2023-08-03
File: sfb_iter.py
"""
import numpy as np
from pathlib import Path
import torch

from trainer import (
    DataLoader,
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


LOGPATH = Path('../log')
LRES = 25
np.set_printoptions(linewidth=160, edgeitems=5, threshold=20,
                    formatter=dict(float=lambda x: f"{x: 9.3e}"))
torch.set_printoptions(linewidth=160, edgeitems=5)


def main(args):
    # ========== Run configuration
    args.logpath = setup_logpath(
        dir=LOGPATH,
        folder_args=(args.data, args.model, args.flag),
        quiet=args.quiet)
    logger = setup_logger(args.logpath, level=args.loglevel, quiet=args.quiet)
    csv_logger = ResLogger(args.logpath.parent.parent, quiet=args.quiet)

    logger.info(f"[args]: {args}")
    csv_logger.concat([
        ('data', args.data),
        ('model', args.model),
        ('seed', args.seed),])
    save_args(args.logpath, args)

    # ========== Load data
    data_loader = DataLoader(args)
    dataset = data_loader(args)

    # ========== Load model
    model_loader = ModelLoader(args)
    model = model_loader(args)

    # ========== Run trainer
    trn = TrnFullbatchIter(
        model=model,
        dataset=dataset,
        args=args,
        logger=logger)
    res_run = trn()

    csv_logger.merge(res_run)
    csv_logger.save()
    logger.log(LRES, f"[res]: {str(csv_logger)}")
    clear_logger(logger)


if __name__ == '__main__':
    parser = setup_argparse()
    # Experiment-specific arguments
    # parser.add_argument()
    args = setup_args(parser)

    main(args)
