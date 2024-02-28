# -*- coding:utf-8 -*-
"""Single run
Author: nyLiao
File Created: 2023-08-03
File: sfb_iter.py
"""
import numpy as np
from pathlib import Path

import torch
import torch_geometric.transforms as T

from torch_geometric.datasets import Planetoid
from torch_geometric.nn.models import GCN

from trainer import (
    LoaderModel,
    TrnFullbatchIter)
from utils import (
    setup_argparse,
    setup_args,
    save_args,
    setup_logger,
    setup_logpath,
    clear_logger,
    ResLogger)


LOGPATH, DATAPATH = Path('../log'), Path('../data')
np.set_printoptions(linewidth=160, edgeitems=5, threshold=20,
                    formatter=dict(float=lambda x: f"{x: 9.3e}"))
torch.set_printoptions(linewidth=160, edgeitems=5)


def main(args):
    # ========== Run configuration
    args.logpath = setup_logpath(
        dir=LOGPATH,
        folder_args=(args.data, args.model, args.flag),
        quiet=args.quiet)
    logger = setup_logger(args.logpath, quiet=args.quiet)
    csv_logger = ResLogger(args.logpath.parent.parent, quiet=args.quiet)

    logger.debug(f"[args]: {args}")
    csv_logger.concat([
        ('data', args.data),
        ('model', args.model),
        ('seed', args.seed),])
    save_args(args.logpath, args)

    # ========== Load data
    # TODO: data loader
    # TODO: general graph norm transform
    dataset = Planetoid(DATAPATH, args.data,
        transform=T.Compose([
            T.NormalizeFeatures(),
            T.ToSparseTensor(),]))

    logger.debug(f"[dataset]: {dataset}")

    # ========== Load model
    # TODO: model loader for checking args.model and kwargs
    # model = GCN(
    #     in_channels=dataset.num_features,
    #     out_channels=dataset.num_classes,
    #     hidden_channels=args.hidden,
    #     num_layers=args.layer,
    #     dropout=args.dp,
    #     normalize=False,
    # ).to(args.device)
    model = LoaderModel(dataset=dataset)(
        model=args.model,
        conv=args.conv,
        args=args).to(args.device)

    logger.debug(f"[model]: {model}")

    # ========== Run trainer
    # TODO: trainer loader
    trn = TrnFullbatchIter(
        model=model,
        dataset=dataset,
        args=args,
        logger=logger)
    res_run = trn()

    csv_logger.merge(res_run)
    csv_logger.save()
    logger.info(f"[res]: {str(csv_logger)}")
    clear_logger(logger)


if __name__ == '__main__':
    parser = setup_argparse()
    # Experiment-specific arguments
    # parser.add_argument()
    args = setup_args(parser)

    main(args)
