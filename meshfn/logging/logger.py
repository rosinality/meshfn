# shamelessly took from detectron2
# https://github.com/facebookresearch/detectron2/blob/master/detectron2/utils/logger.py

import functools
import logging
import sys
import pprint
import inspect
from typing import List, Optional
import os

try:
    from termcolor import colored

except ImportError:
    colored = None

try:
    from rich.logging import RichHandler

except ImportError:
    RichHandler = None

from meshfn.distributed.parallel_mode import ParallelMode


class ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super().__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super().formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


def wrap_log_record_factory(factory):
    def wrapper(
        name, level, fn, lno, msg, args, exc_info, func=None, sinfo=None, **kwargs
    ):
        if not isinstance(msg, str):
            msg = pprint.pformat(msg)

        return factory(name, level, fn, lno, msg, args, exc_info, func, sinfo, **kwargs)

    return wrapper


class DistributedLogger:
    def __init__(
        self, name, parallel_context=None, mode="rich", abbrev_name=None, keywords=None
    ):
        self.logger = make_logger(
            name, mode=mode, abbrev_name=abbrev_name, keywords=keywords
        )
        self.parallel_context = parallel_context

    @staticmethod
    def get_call_info():
        stack = inspect.stack()

        fn = stack[3][1]
        ln = stack[3][2]
        func = stack[3][3]

        return os.path.basename(fn), ln, func

    def log(
        self,
        level,
        message,
        parallel_mode: ParallelMode.GLOBAL,
        ranks: Optional[List[int]] = None,
    ):
        if ranks is None:
            return getattr(self.logger, level)(message)

        local_rank = self.parallel_context.local_rank(parallel_mode)

        if local_rank in ranks:
            return getattr(self.logger, level)(message)

    def message_prefix(self):
        return "FROM {}:{} {}()".format(*self.get_call_info())

    def info(
        self,
        message: str,
        parallel_mode: ParallelMode = ParallelMode.GLOBAL,
        ranks: Optional[List[int]] = None,
    ):
        self.log("info", self.message_prefix(), parallel_mode, ranks)
        self.log("info", message, parallel_mode, ranks)

    def warning(
        self,
        message: str,
        parallel_mode: ParallelMode = ParallelMode.GLOBAL,
        ranks: Optional[List[int]] = None,
    ):
        self.log("warning", self.message_prefix(), parallel_mode, ranks)
        self.log("warning", message, parallel_mode, ranks)

    def debug(
        self,
        message: str,
        parallel_mode: ParallelMode = ParallelMode.GLOBAL,
        ranks: Optional[List[int]] = None,
    ):
        self.log("debug", self.message_prefix(), parallel_mode, ranks)
        self.log("debug", message, parallel_mode, ranks)

    def error(
        self,
        message: str,
        parallel_mode: ParallelMode = ParallelMode.GLOBAL,
        ranks: Optional[List[int]] = None,
    ):
        self.log("error", self.message_prefix(), parallel_mode, ranks)
        self.log("error", message, parallel_mode, ranks)


@functools.lru_cache()  # so that calling setup_logger multiple times won't add many handlers
def make_logger(name="main", mode="rich", abbrev_name=None, keywords=None):
    """
    Initialize the detectron2 logger and set its verbosity level to "DEBUG".
    Args:
        output (str): a file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        name (str): the root module name of this logger
        abbrev_name (str): an abbreviation of the module, to avoid long names in logs.
            Set to "" to not log the root module in logs.
            By default, will abbreviate "detectron2" to "d2" and leave other
            modules unchanged.
    Returns:
        logging.Logger: a logger
    """

    logging.setLogRecordFactory(wrap_log_record_factory(logging.getLogRecordFactory()))

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if abbrev_name is None:
        abbrev_name = name

    if mode == "rich" and RichHandler is None:
        mode = "color"

    if mode == "color" and colored is None:
        mode = "plain"

    if mode == "color":
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = ColorfulFormatter(
            colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
            datefmt="%m/%d %H:%M:%S",
            root_name=name,
            abbrev_name=str(abbrev_name),
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    elif mode == "rich":
        logger.addHandler(
            RichHandler(
                level=logging.DEBUG, log_time_format="%m/%d %H:%M:%S", keywords=keywords
            )
        )

    elif mode == "plain":
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
            datefmt="%m/%d %H:%M:%S",
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger
