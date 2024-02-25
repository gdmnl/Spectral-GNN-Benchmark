# -*- coding:utf-8 -*-
"""
Author: nyLiao
File Created: 2023-03-20
File: logger.py
"""
from typing import Tuple
from pathlib import Path
from logging import Logger

import sys
import logging
import uuid
import pandas as pd


def setup_logger(filename: Path = Path('log.txt'),
                 level=logging.DEBUG,
                 fmt='{message}'):
    logger = logging.getLogger(filename.stem)
    logger.logpath = filename.parent

    formatter = logging.Formatter(fmt, style='{')
    fileHandler = logging.FileHandler(filename, delay=True)
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler(stream=sys.stdout)
    streamHandler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)
    return logger


def clear_logger(logger: Logger):
    handlers = logger.handlers[:]
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()


def setup_logpath(dir: Path = Path('log'),
                  folder_args: Tuple=None,
                  name_args: Tuple=None):
    if folder_args is not None:
        dir = dir.joinpath(*folder_args)
    else:
        flag = str(uuid.uuid4())[:6]
        dir = dir.joinpath(flag)
    dir.mkdir(parents=True, exist_ok=True)

    if name_args is None:
        return dir
    fname = '-'.join(name_args) + '.txt'
    return dir.joinpath(fname)


class CSVLogger(object):
    def __init__(self, filename: Path = Path('log/summary.csv')):
        self.filename = filename
        self.data = pd.DataFrame()

    def _guess_fmt(self, col: str, val):
        if isinstance(val, str):
            return (lambda x: x)
        if isinstance(val, int):
            return (lambda x: format(x, 'd'))
        if isinstance(val, float):
            if col.count('acc'):
                return (lambda x: format(x, '.4f'))
            if col.count('loss'):
                return (lambda x: format(x, '.6f'))
            if col.count('time'):
                return (lambda x: format(x, '.4f'))
            if col.count('mem') or col.count('num'):
                return (lambda x: format(x, '.3f'))

    def _log_single(self, val,
            col: str, row: int = 0,
            fmt = None):
        fmt = fmt or self._guess_fmt(col, val)
        if col not in self.data.columns:
            self.data[col] = None
        self.data.loc[row, col] = val
        self.data[col] = self.data[col].apply(fmt)

    def log(self, vals: dict, row: int = 0):
        for col, val in vals.items():
            self._log_single(val, col, row)

    def print(self):
        self.filename.parent.mkdir(parents=True, exist_ok=True)
        with open(self.filename, 'a') as f:
            # TODO: manage column difference in header
            self.data.to_csv(self.filename, index=False,
                             mode='a', header=f.tell()==0)
