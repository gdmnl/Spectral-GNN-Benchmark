# -*- coding:utf-8 -*-
"""
Author: nyLiao
File Created: 2023-03-20
File: logger.py
"""
from typing import Tuple, List, Callable, Union, Any
from pathlib import Path
from logging import Logger

import os
import sys
import logging
import uuid
import pandas as pd
from pandas import DataFrame, Series


LOGPATH = Path('../log')


def setup_logger(logpath: Union[Path, str] = LOGPATH,
                 level=logging.DEBUG,
                 quiet: bool = True,
                 fmt='{message}'):
    logpath = Path(logpath)
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

    # FEATURE: [wandb](https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/logging.py)

    return logger


def clear_logger(logger: Logger):
    handlers = logger.handlers[:]
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()


def setup_logpath(dir: Union[Path, str] = LOGPATH,
                  folder_args: Tuple=None,
                  name_args: Tuple=None,
                  quiet: bool = True):
    r"""Resolve log path for saving.

    Args:
        dir (Path or str): Base directory for saving logs. Default is './log/'.
        folder_args (Tuple): Subfolder names.
        name_args (Tuple): File name components.
        quiet (bool, optional): Quiet run without creating directories.

    Returns:
        logpath (Path): Path for log file/directory.
    """
    dir = Path(dir)
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


class ResLogger(object):
    r"""Logger for formatting table data to strings by wrapping pd.DataFrame.

    Args:
        logpath (Path or str): Path to CSV file saving directory.
        quiet (bool): Quiet run without saving file.
    """
    def __init__(self,
                 logpath: Union[Path, str] = LOGPATH,
                 quiet: bool = True):
        logpath = Path(logpath)
        self.filename = logpath.joinpath('summary.csv')
        self.quiet = quiet
        self.data = DataFrame()
        self.fmt = Series()

    @staticmethod
    def guess_fmt(key: str, val) -> Callable:
        """Guesses the string format function based on its name.
        """
        if isinstance(val, str) or isinstance(val, bool):
            return (lambda x: x)
        if isinstance(val, int):
            if key.startswith(('epoch', 'iter')):
                return (lambda x: format(x, '03d'))
            return (lambda x: format(x, 'd'))
        if isinstance(val, float):
            if key.startswith(('acc', 'metric', 'score')):
                return (lambda x: format(x, '.4f'))
            if key.startswith(('loss',)):
                return (lambda x: format(x, '.6f'))
            if key.startswith(('time',)):
                return (lambda x: format(x, '.4f'))
            if key.startswith(('mem', 'num')):
                return (lambda x: format(x, '.3f'))
            return (lambda x: format(x, '.3e'))
        return (lambda x: str(x))

    @property
    def nrows(self):
        return self.data.shape[0]

    @property
    def ncols(self):
        return self.data.shape[1]

    def __getitem__(self, key):
        return self.data.loc[key]

    def __str__(self) -> str:
        return self.get_str(maxlen=80)

    # ===== Input
    def _set(self, data: DataFrame, fmt: Series):
        # for coli in list(set(data.columns) - set(self.data.columns)):
        #     self.data[coli] = None
        #     self.fmt[coli] = None

        if self.data.empty:
            self.data = data
            self.fmt = fmt
        else:
            self.data = pd.concat([self.data, data], axis=1, join='inner', copy=False)
            self.fmt = pd.concat([self.fmt, fmt], axis=0, join='inner', copy=False)

    def concat(self,
               vals: List[Tuple[str, Any, Callable]],
               row: int = 0,
               suffix: str = None):
        r"""Concatenate data entries of a single row to data.

        Args:
            vals (List): list of entries (key, value, formatter).
            row (int): New index in self dataframe for vals to be logged.
            suffix (str): Suffix string for input keys. Default is None.
        """
        val_dct, fmt_dct = {}, {}
        for vali in vals:
            if len(vali) == 2:
                col, val = vali
                fmt = None
            else:
                col, val, fmt = vali
            col = f'{col}_{suffix}' if suffix else col

            val_dct[col] = val
            fmt_dct[col] = fmt

        self._set(pd.DataFrame(val_dct, index=[row]), Series(fmt_dct))
        return self

    def merge(self,
              logger: 'ResLogger',
              rows: List[int] = None,
              suffix: str = None):
        r"""Merge from another logger.

        Args:
            vals (TabLogger): Logger to merge.
            row (List): New index in self dataframe.
            suffix (str): Suffix string for input keys. Default is None.
        """
        if rows:
            assert len(rows) == logger.nrows
            logger.data.index = rows
        if suffix:
            logger.data.columns = [f'{coli}_{suffix}' for coli in logger.data.columns]

        self._set(logger.data, logger.fmt)
        return self

    # ===== Output
    def _get(self, col = None, row = None) -> Union[DataFrame, Series, str]:
        row = row or self.data.index
        col = col or self.data.columns
        mrow, mcol = isinstance(row, list), isinstance(col, list)
        val = self.data.loc[row, col]

        if mcol:
            # Return DataFrame
            for coli in col:
                fmt = self.fmt[coli]
                fmt = fmt if callable(fmt) else self.guess_fmt(coli, val[coli].iloc[0])
                val[coli] = val[coli].map(fmt)
            return val
        elif mrow:
            # Return Series
            fmt = self.fmt[col]
            fmt = fmt if callable(fmt) else self.guess_fmt(col, val.iloc[0])
            return val.map(fmt)
        else:
            # Return scalar str
            val = val.item() if isinstance(val, Series) else val
            fmt = self.fmt[col]
            fmt = fmt if callable(fmt) else self.guess_fmt(col, val)
            return fmt(val)

    def save(self):
        r"""Saves table data to CSV file.
        """
        if self.quiet:
            return
        data_str = DataFrame()
        for coli in self.data.columns:
            data_str[coli] = self._get(col=coli)
        with open(self.filename, 'a') as f:
            # FEATURE: manage column difference in header
            data_str.to_csv(self.filename, index=False,
                             mode='a', header=f.tell()==0)

    def get_str(self,
                col: Union[List, str] = None,
                row: Union[List, int] = None,
                maxlen: int = -1) -> str:
        if col is None:
            col = self.data.columns
        elif isinstance(col, str):
            col = [col]
        if row is None:
            if self.nrows > 1:
                row = self.data.index.tolist()
            else:
                row = [self.data.index[0]]
        elif isinstance(row, int):
            if row < 0:
                row = self.nrows + row
            row = [row]

        result, length = [], 0
        for coli in col:
            if len(row) > 1:
                resstr = ','.join(self._get(coli, row).tolist())
                resstr = f'{coli}:({resstr})'
            else:
                resstr = f'{coli}:{self._get(coli, row[0])}'

            length += len(resstr)
            if maxlen > 0 and length > maxlen:
                resstr = '\n' + resstr
                length = 0
            result.append(resstr)
        return ', '.join(result)
