# -*- coding:utf-8 -*-
"""
Author: nyLiao
File Created: 2023-03-20
File: logger.py
"""
from typing import Tuple
from pathlib import Path
from logging import Logger

import os
import sys
import logging
import uuid
import pandas as pd


def setup_logger(logpath: Path,
                 level=logging.DEBUG,
                 quiet: bool = True,
                 fmt='{message}'):
    formatter = logging.Formatter(fmt, style=fmt[0])
    logger = logging.getLogger('log')
    logger.setLevel(level)

    streamHandler = logging.StreamHandler(stream=sys.stdout)
    streamHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)

    if not quiet:
        filename = logpath.joinpath('log.txt')
        if os.path.exists(filename):
            logger.warning(f'Warning: Log file {filename.absolute()} already exists, will be overwritten.')
            os.remove(filename)

        fileHandler = logging.FileHandler(filename, delay=True)
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

    # TODO: [wandb](https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/logging.py)

    return logger


def clear_logger(logger: Logger):
    handlers = logger.handlers[:]
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()


def setup_logpath(dir: Path = Path('log'),
                  folder_args: Tuple=None,
                  name_args: Tuple=None,
                  quiet: bool = True):
    r"""Resolve log path for saving.

    Args:
        dir (Path): The base directory for saving logs. Default is './log/'.
        folder_args (Tuple): Subfolder names.
        name_args (Tuple): File name components.
        quiet (bool, optional): Quiet run without creating directories.

    Returns:
        logpath (Path): Path for log file/directory.
    """
    if folder_args is not None:
        dir = dir.joinpath(*folder_args)
    else:
        flag = str(uuid.uuid4())[:6]
        dir = dir.joinpath(flag)
    if not quiet:
        dir.mkdir(parents=True, exist_ok=True)

    if name_args is None:
        return dir
    fname = '-'.join(name_args) + '.txt'
    return dir.joinpath(fname)


class CSVLogger(object):
    r"""A class for logging data to a CSV file.

    Args:
        logpath (Path): Path to CSV file saving directory.
        quiet (bool): Quiet run without saving file.
    """
    def __init__(self, logpath: Path, quiet: bool = True):
        self.filename = logpath.joinpath('summary.csv')
        self.quiet = quiet
        self.data = pd.DataFrame()

    def _guess_fmt(self, key: str, val):
        """Guesses the format function for a given column based on its value type.
        """
        if isinstance(val, str) or isinstance(val, bool):
            return (lambda x: x)
        if isinstance(val, int):
            return (lambda x: format(x, 'd'))
        if isinstance(val, float):
            if key.count('acc') or key.count('score'):
                return (lambda x: format(x, '.4f'))
            if key.count('loss'):
                return (lambda x: format(x, '.6f'))
            if key.count('time'):
                return (lambda x: format(x, '.4f'))
            if key.count('mem') or key.count('num'):
                return (lambda x: format(x, '.3f'))

    def _log_single(self, val,
            col: str, row: int = 0,
            fmt = None):
        """Log a single value to the specified column and row.
        """
        fmt = fmt or self._guess_fmt(col, val)
        if col not in self.data.columns:
            self.data[col] = None
        self.data.loc[row, col] = val
        self.data[col] = self.data[col].apply(fmt)

    def log(self, vals: Tuple, row: int = 0):
        r"""Log data entries.

        Args:
            vals (Tuple): Tuple of (key, value, formatter).
            row (int): The row index. Default is 0.
        """
        for vali in vals:
            if len(vali) == 2:
                key, val = vali
                fmt = None
            else:
                key, val, fmt = vali
            self._log_single(val, key, row, fmt)

    def save(self):
        r"""Saves the logged data to the CSV file.
        """
        if self.quiet:
            return
        with open(self.filename, 'a') as f:
            # TODO: manage column difference in header
            self.data.to_csv(self.filename, index=False,
                             mode='a', header=f.tell()==0)

    def __str__(self) -> str:
        result = []
        length = 0
        for col in self.data.columns:
            if len(self.data[col]) > 1:
                resstr = f'{col}:{self.data[col].values.tolist()}'
            else:
                resstr = f'{col}:{self.data[col].values[0]}'
            length += len(resstr)
            if length > 80:
                resstr = '\n' + resstr
                length = 0
            result.append(resstr)
        return ', '.join(result)
