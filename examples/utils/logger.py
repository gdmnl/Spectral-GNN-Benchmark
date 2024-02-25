# -*- coding:utf-8 -*-
"""
Author: nyLiao
File Created: 2023-03-20
File: logger.py
"""
import sys
import logging
import uuid

from typing import Tuple
from pathlib import Path
from logging import Logger


def setup_logger(filename: Path = Path('log.txt'),
                 level=logging.DEBUG, fmt='{message}'):
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
