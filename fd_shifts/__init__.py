from omegaconf import OmegaConf

from .version import version
OmegaConf.register_new_resolver("fd_shifts.version", version)
