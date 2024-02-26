# -*- coding:utf-8 -*-
"""Single model, full batch, iterative model
Author: nyLiao
File Created: 2023-08-03
File: sfb_iter.py
"""
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn.models import GCN
from pyg_spectral import metrics

from utils import (
    setup_argparse,
    setup_args,
    save_args,
    setup_logger,
    setup_logpath,
    clear_logger,
    CSVLogger,
    CkptLogger)


LOGPATH, DATAPATH = Path('../log'), Path('../data')
np.set_printoptions(linewidth=160, edgeitems=5, threshold=20,
                    formatter=dict(float=lambda x: "% 9.3e" % x))
torch.set_printoptions(linewidth=160, edgeitems=5)


def main(args):
    # ========== Run configuration
    args.logpath = setup_logpath(
        dir=LOGPATH,
        folder_args=(args.data, args.model, args.flag),
        quiet=args.quiet)
    logger = setup_logger(args.logpath, quiet=args.quiet)
    csv_logger = CSVLogger(args.logpath.parent.parent, quiet=args.quiet)
    # assert (args.quiet or '_file' not in storage), "Storage scheme cannot be file for quiet run."
    ckpt_logger = CkptLogger(
        args.logpath,
        prefix=(f'model-{args.suffix}' if args.suffix else 'model'),
        patience=args.patience, period=args.period, storage='state_gpu')

    logger.debug(f"[args]:{args}")
    save_args(args.logpath, args)

    # ========== Load data
    # TODO: data parser
    # TODO: general graph norm transform
    dataset = Planetoid(DATAPATH, args.data,
        transform=T.Compose([T.NormalizeFeatures(),
                             T.ToSparseTensor()]))
    data = dataset[0]

    logger.debug(f"[dataset]:{dataset}")
    logger.debug(f"[data]:{data}")

    # ========== Load model
    # TODO: model parser
    model = GCN(
        in_channels=dataset.num_features,
        out_channels=dataset.num_classes,
        hidden_channels=args.hidden,
        num_layers=args.layer,
        dropout=args.dp,
        normalize=False,
    ).to(args.device)

    logger.debug(f"[model]:{model}")

    # TODO: layer-specific wd (then enable scheduler)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='max', factor=0.5,
    #     threshold=1e-4, patience=15, verbose=False)
    loss_fn = nn.CrossEntropyLoss()
    stopwatch = metrics.Stopwatch()

    # ========== Trainer
    def train_epoch(x, edge_idx, y, split_mask):
        model.train()
        x, y = x.to(args.device), y[split_mask].to(args.device)
        edge_idx = edge_idx.to(args.device)
        stopwatch.reset()

        stopwatch.start()
        optimizer.zero_grad()
        output = model(x, edge_idx)[split_mask]
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        stopwatch.pause()

        return loss.item(), stopwatch.time

    def eval_epoch(x, edge_idx, y, split_mask):
        model.eval()
        x, y = x.to(args.device), y[split_mask].to(args.device)
        edge_idx = edge_idx.to(args.device)
        calc = metrics.F1Calculator(dataset.num_classes)
        stopwatch.reset()

        with torch.no_grad():
            stopwatch.start()
            output = model(x, edge_idx)[split_mask]
            stopwatch.pause()

            output = output.cpu().detach()
            output = output.argmax(dim=1)
            ylabel = y.cpu().detach()
            calc.update(ylabel, output)

        res = calc.get('micro')
        return res, stopwatch.time

    # ========== Train
    with args.device:
        torch.cuda.empty_cache()
    time_train = metrics.Accumulator()

    for epoch in range(1, args.epoch+1):
        loss_train, time_epoch = train_epoch(
            x=data.x,
            edge_idx=data.adj_t,
            y=data.y,
            split_mask=data.train_mask)
        time_train.update(time_epoch)
        acc_val, _ = eval_epoch(
            x=data.x,
            edge_idx=data.adj_t,
            y=data.y,
            split_mask=data.val_mask)
        # scheduler.step(acc_val)

        logstr = f"Epoch:{epoch:04d} | loss_train:{loss_train:.4f}, acc_val:{acc_val:.4f}, time_train:{time_train.data:.4f}"
        logger.info(logstr)

        early_stop = ckpt_logger.step(acc_val, model)
        if early_stop:
            break

    csv_logger.log((
        ('epoch_total', ckpt_logger.epoch_current),
        ('epoch_best', ckpt_logger.epoch_best),
        ('time_train', time_train.data),
        ('acc_val', acc_val)))

    # ========== Test
    model = ckpt_logger.load('best', model=model)
    acc_test, time_test = eval_epoch(
        x=data.x,
        edge_idx=data.adj_t,
        y=data.y,
        split_mask=data.test_mask)
    mem_rem = metrics.MemoryRAM().get(unit='G')
    mem_cuda = metrics.MemoryCUDA().get(unit='G')

    csv_logger.log((
        ('acc_test', acc_test),
        ('time_test', time_test),
        ('mem_rem', mem_rem),
        ('mem_cuda', mem_cuda)))
    csv_logger.save()
    logger.info(str(csv_logger))
    clear_logger(logger)


if __name__ == '__main__':
    parser = setup_argparse()
    # Experiment-specific arguments
    # parser.add_argument()
    args = setup_args(parser)

    main(args)
