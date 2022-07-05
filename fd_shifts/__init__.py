import logging
from random import randint

from loguru import logger
from omegaconf import OmegaConf

from .version import version


class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame: logging.FrameType = logging.currentframe()
        depth: int = 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


logging.basicConfig(handlers=[InterceptHandler()], level=logging.DEBUG)

OmegaConf.register_new_resolver("fd_shifts.version", version)
OmegaConf.register_new_resolver("fd_shifts.random_seed", lambda: randint(0, 1_000_000))
