import logging
import warnings
from functools import reduce
from random import randint
from types import FrameType
from typing import TypeVar

from loguru import logger
from omegaconf import OmegaConf

from .version import get_version

warnings.filterwarnings(
    "ignore", "You want to use `wandb` which is not installed yet", UserWarning
)
warnings.filterwarnings(
    "ignore", "You want to use `gym` which is not installed yet", UserWarning
)
warnings.filterwarnings(
    "ignore", "fields may not start with an underscore", RuntimeWarning
)
warnings.filterwarnings(
    "ignore",
    "The value of the smallest subnormal for",
    UserWarning,
)


class InterceptHandler(logging.Handler):
    """Replace python logging everywhere"""

    def emit(self, record):  # type: ignore
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame: FrameType = logging.currentframe()
        depth: int = 2
        while frame.f_code.co_filename == logging.__file__:
            if (_frame := frame.f_back) is None:
                break
            frame = _frame
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


logging.basicConfig(handlers=[InterceptHandler()], level=logging.WARNING)

OmegaConf.register_new_resolver("fd_shifts.version", get_version)
OmegaConf.register_new_resolver("fd_shifts.random_seed", lambda: randint(0, 1_000_000))
OmegaConf.register_new_resolver(
    "fd_shifts.if_else", lambda cond, a, b: a if cond else b
)
OmegaConf.register_new_resolver(
    "fd_shifts.ifeq_else", lambda cond, x, a, b: a if cond == x else b
)

T = TypeVar("T")


def _concat(*args: list[T]) -> list[T]:
    return reduce(lambda a, b: a + b, args)


OmegaConf.register_new_resolver("fd_shifts.concat", _concat)
