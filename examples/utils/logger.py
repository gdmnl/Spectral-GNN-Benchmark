# -*- coding:utf-8 -*-
"""
Author: nyLiao
File Created: 2023-03-20
"""
from typing import Tuple, List, Dict, Callable, Union, Any
from pathlib import Path

import os
import sys
import logging
import uuid
from datetime import datetime
import numpy as np
import pandas as pd
from pandas import DataFrame, Series


LOGPATH = Path('../log')


def setup_logger(logpath: Union[Path, str] = LOGPATH,
                 level: int = logging.DEBUG,
                 quiet: bool = True,
                 fmt='{message}'):
    logging.LTRN = 15
    logging.LRES = 25
    logging.addLevelName(logging.LTRN, 'TRAIN')
    logging.addLevelName(logging.LRES, 'RESULT')
    logpath = Path(logpath)
    formatter = logging.Formatter(fmt, style=fmt[0])
    logger = logging.getLogger('log')
    logger.setLevel(logging.DEBUG)

    consoleHandler = logging.StreamHandler(stream=sys.stdout)
    consoleHandler.setFormatter(formatter)
    consoleHandler.setLevel(level)
    logger.addHandler(consoleHandler)

    if not quiet:
        filename = logpath.joinpath('log.txt')
        if os.path.exists(filename):
            logger.warning(f'Warning: Log file {filename.resolve()} already exists, will be overwritten.')
            os.remove(filename)

        fileHandler = logging.FileHandler(filename, delay=True)
        fileHandler.setFormatter(formatter)
        fileHandler.setLevel(logging.LTRN)
        logger.addHandler(fileHandler)

    logger.log(logging.LTRN, f"[time]: {datetime.now()}")
    return logger


def clear_logger(logger: logging.Logger):
    logger.log(logging.LTRN, f"[time]: {datetime.now()}")
    handlers = logger.handlers[:]
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()
    logger.info(f"[time]: {datetime.now()}")


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
    r"""Logger for formatting result to strings by wrapping pd.DataFrame table.

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
        if isinstance(val, str):
            return (lambda x: x)
        if np.issubdtype(type(val), np.integer):
            return (lambda x: format(x, 'd'))
        if np.issubdtype(type(val), np.floating):
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
        r"""Short for pandas.DataFrame.loc()"""
        return self.data.loc[key]

    # ===== Input
    def _set(self, data: DataFrame, fmt: Series):
        r"""Sets the data from input DataFrame.

        Args:
            data (DataFrame): Concat on columns, inner join on index.
            fmt (Series): Inner join on columns.
        """
        cols_left = self.data.columns.tolist()
        cols_right = data.columns.tolist()
        cols = list(dict.fromkeys(cols_left + cols_right))

        self.data = self.data.combine_first(data)
        self.data = self.data.reindex(cols, axis=1)
        self.fmt = self.fmt.combine_first(fmt)

    def concat(self,
               vals: Union[List[Tuple[str, Any, Callable]], Dict],
               row: int = 0,
               suffix: str = None):
        r"""Concatenate data entries of a single row to data.

        Args:
            vals (List or Dict): list of entries (key, value, formatter).
            row (int): New index in self dataframe for vals to be logged.
            suffix (str): Suffix string for input keys. Default is None.

        Returns:
            self (ResLogger)
        """
        val_dct, fmt_dct = {}, {}
        if isinstance(vals, list):
            for vali in vals:
                if len(vali) == 2:
                    col, val = vali
                    fmt = None
                else:
                    col, val, fmt = vali
                col = f'{col}_{suffix}' if suffix else col

                val_dct[col] = val
                fmt_dct[col] = fmt
        elif isinstance(vals, dict):
            for col, val in vals.items():
                col = f'{col}_{suffix}' if suffix else col
                val_dct[col] = val
                fmt_dct[col] = None

        self._set(pd.DataFrame(val_dct, index=[row]), Series(fmt_dct))
        return self

    def __call__(self, *args, **kwargs) -> 'ResLogger':
        r"""Short for concat()"""
        return self.concat(*args, **kwargs)

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

    def del_col(self, col: Union[List, str]) -> 'ResLogger':
        r"""Delete columns from data.

        Args:
            col (str or list): Column(s) to delete.
        """
        self.data = self.data.drop(columns=col)
        self.fmt = self.fmt.drop(index=col)
        return self

    # ===== Output
    def _get(self, col=None, row=None) -> Union[DataFrame, Series, str]:
        r"""Retrieve one or sliced data and apply string format.

        Args:
            col (str or list): Column(s) to retrieve. Defaults to all.
            row (str or list): Row(s) to retrieve. Defaults to all.

        Returns:
            val: Formatted data.
                - type: follows the return type of DataFrame.loc[row, col].
                - value: formatted string in each entry.
        """
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
        data_str = DataFrame(columns=self.data.columns, index=[0])
        for coli in self.data.columns:
            data_str.loc[0, coli] = self._get(col=coli)
        with open(self.filename, 'a') as f:
            # FEATURE: manage column inconsistency in header
            data_str.to_csv(self.filename, index=False,
                             mode='a', header=f.tell()==0)

    def get_str(self,
                col: Union[List, str] = None,
                row: Union[List, int] = None,
                maxlen: int = -1) -> str:
        r"""Get formatted long string for printing of the specified columns
        and rows.

        Args:
            col (str or list): Column(s) to retrieve. Defaults to all.
            row (str or list): Row(s) to retrieve. Defaults to all.
            maxlen (int): Max line length of the resulting string.

        Returns:
            s (str): Formatted string representation.
        """
        if self.data.empty:
            return ''
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
                resstr = '\n  ' + resstr
                length = 0
            result.append(resstr)
        return ', '.join(result)

    def __str__(self) -> str:
        r"""String for print on screen."""
        return self.get_str(maxlen=80)
