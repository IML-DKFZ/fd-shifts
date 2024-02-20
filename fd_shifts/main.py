from __future__ import annotations

import types
import typing
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

import jsonargparse
import rich
import yaml
from jsonargparse import ActionConfigFile, ArgumentParser
from jsonargparse._actions import Action
from omegaconf import OmegaConf
from rich.pretty import pretty_repr

from fd_shifts.configs import Config

__subcommands = {}


def subcommand(func: Callable):
    __subcommands[func.__name__] = func
    return func


previous_config: ContextVar = ContextVar("previous_config", default=None)


@contextmanager
def previous_config_context(cfg):
    token = previous_config.set(cfg)
    try:
        yield
    finally:
        previous_config.reset(token)


class ActionExperiment(Action):
    """Action to indicate that an argument is an experiment name."""

    def __init__(self, **kwargs):
        """Initializer for ActionExperiment instance."""
        if "default" in kwargs:
            raise ValueError("ActionExperiment does not accept a default.")
        opt_name = kwargs["option_strings"]
        opt_name = (
            opt_name[0]
            if len(opt_name) == 1
            else [x for x in opt_name if x[0:2] == "--"][0]
        )
        if "." in opt_name:
            raise ValueError("ActionExperiment must be a top level option.")
        if "help" not in kwargs:
            # TODO: hint to list-experiments
            kwargs["help"] = "Name of an experiment."
        super().__init__(**kwargs)

    def __call__(self, parser, cfg, values, option_string=None):
        """Parses the given experiment configuration and adds all the corresponding keys to the namespace.

        Raises:
            TypeError: If there are problems parsing the configuration.
        """
        self.apply_experiment_config(parser, cfg, self.dest, values)

    @staticmethod
    def apply_experiment_config(parser: ArgumentParser, cfg, dest, value) -> None:
        from fd_shifts.experiments.configs import get_experiment_config

        with previous_config_context(cfg):
            experiment_cfg = get_experiment_config(value)
            tcfg = parser.parse_object(
                {"config": asdict(experiment_cfg)},
                env=False,
                defaults=False,
                _skip_check=True,
            )
            cfg_merged = parser.merge_config(tcfg, cfg)
            cfg.__dict__.update(cfg_merged.__dict__)
            cfg[dest] = value


class ActionLegacyConfigFile(ActionConfigFile):
    """Action to indicate that an argument is a configuration file or a configuration string."""

    def __init__(self, **kwargs):
        """Initializer for ActionLegacyConfigFile instance."""
        if "default" in kwargs:
            self.set_default_error()
        opt_name = kwargs["option_strings"]
        opt_name = (
            opt_name[0]
            if len(opt_name) == 1
            else [x for x in opt_name if x[0:2] == "--"][0]
        )
        if "." in opt_name:
            raise ValueError("ActionLegacyConfigFile must be a top level option.")
        if "help" not in kwargs:
            kwargs["help"] = "Path to a configuration file."
        super().__init__(**kwargs)

    def __call__(self, parser, cfg, values, option_string=None):
        """Parses the given configuration and adds all the corresponding keys to the namespace.

        Raises:
            TypeError: If there are problems parsing the configuration.
        """
        self.apply_config(parser, cfg, self.dest, values, option_string)

    @staticmethod
    def set_default_error():
        raise ValueError(
            "ActionLegacyConfigFile does not accept a default, use default_config_files."
        )

    @staticmethod
    def apply_config(parser, cfg, dest, value, option_string) -> None:
        from jsonargparse._link_arguments import skip_apply_links

        from fd_shifts.experiments.configs import get_dataset_config

        with jsonargparse._actions._ActionSubCommands.not_single_subcommand(), previous_config_context(
            cfg
        ), skip_apply_links():
            kwargs = {
                "env": False,
                "defaults": False,
                "_skip_check": True,
            }
            cfg_path: Optional[jsonargparse.Path] = jsonargparse.Path(
                value, mode=jsonargparse._optionals.get_config_read_mode()
            )

            with cfg_path.open() as f:
                cfg_from_file = yaml.unsafe_load(f)

            if option_string == "--config-file":
                cfg_file = cfg_from_file
            elif option_string == "--legacy-config-file":
                cfg_file = {"config": cfg_from_file}

                # hydra instantiate to jsonargparse instantiate format
                lr_scheduler_cfg = cfg_file["config"]["trainer"]["lr_scheduler"]
                cfg_file["config"]["trainer"]["lr_scheduler"] = {
                    "class_path": "fd_shifts.configs.LRSchedulerConfig",
                    "init_args": {
                        "class_path": lr_scheduler_cfg["_target_"],
                        "init_args": {
                            k: v
                            for k, v in lr_scheduler_cfg.items()
                            if k not in ["_target_", "_partial_"]
                        },
                    },
                }
                optimizer_cfg = cfg_file["config"]["trainer"]["optimizer"]
                cfg_file["config"]["trainer"]["optimizer"] = {
                    "class_path": "fd_shifts.configs.OptimizerConfig",
                    "init_args": {
                        "class_path": optimizer_cfg["_target_"],
                        "init_args": {
                            k: v
                            for k, v in optimizer_cfg.items()
                            if k not in ["_target_", "_partial_"]
                        },
                    },
                }

                # query_studies contain DataConfig objects now, not just names
                for k, v in cfg_file["config"]["eval"]["query_studies"].items():
                    if k == "iid_study":
                        pass
                    elif k == "noise_study":
                        if len(v) == 0:
                            cfg_file["config"]["eval"]["query_studies"][k] = None
                        elif len(v) == 1:
                            cfg_file["config"]["eval"]["query_studies"][k] = asdict(
                                get_dataset_config(v[0])
                            )
                        else:
                            raise ValueError(f"Too many noise studies {v}")
                    elif k in ["in_class_study", "new_class_study"]:
                        cfg_file["config"]["eval"]["query_studies"][k] = [
                            asdict(get_dataset_config(v2)) for v2 in v
                        ]
                    else:
                        raise ValueError(f"Unknown query study {k}")

                # for specific experiments, the seed should be fixed, if "random_seed" was written fix it
                if isinstance(cfg_file["config"]["exp"]["global_seed"], str):
                    warnings.warn(
                        "global_seed is set to random in file, setting it to -1"
                    )
                    cfg_file["config"]["exp"]["global_seed"] = -1

                # hydra is gone
                if cfg_file["config"]["exp"]["work_dir"] == "${hydra:runtime.cwd}":
                    cfg_file["config"]["exp"]["work_dir"] = Path.cwd()

                # resolve everything else
                oc_config = OmegaConf.create(cfg_file["config"])
                dict_config: dict[str, Any] = OmegaConf.to_object(oc_config)  # type: ignore
                cfg_file["config"] = dict_config

                # don't need to comply with accumulate_grad_batches, that's runtime env dependent
                cfg_file["config"]["trainer"]["batch_size"] *= cfg_file["config"][
                    "trainer"
                ].get("accumulate_grad_batches", 1)
                cfg_file["config"]["trainer"]["accumulate_grad_batches"] = 1

            else:
                raise ValueError(f"Unknown option string {option_string}")

            cfg_file = parser.parse_object(cfg_file, **kwargs)

            cfg_merged = parser.merge_config(cfg_file, cfg)
            cfg.__dict__.update(cfg_merged.__dict__)
            if cfg.get(dest) is None:
                cfg[dest] = []
            cfg[dest].append(cfg_path)


def _path_to_str(cfg) -> dict:
    def __path_to_str(cfg):
        if isinstance(cfg, dict):
            return {k: __path_to_str(v) for k, v in cfg.items()}
        if is_dataclass(cfg):
            return cfg.__class__(
                **{k: __path_to_str(v) for k, v in cfg.__dict__.items()}
            )
        if isinstance(cfg, list):
            return [__path_to_str(v) for v in cfg]
        if isinstance(cfg, Path):
            return str(cfg)
        return cfg

    return __path_to_str(cfg)  # type: ignore


def _dict_to_dataclass(cfg) -> Config:
    def __dict_to_dataclass(cfg, cls):
        if is_dataclass(cls):
            fieldtypes = typing.get_type_hints(cls)
            return cls(
                **{k: __dict_to_dataclass(v, fieldtypes[k]) for k, v in cfg.items()}
            )
        if (
            isinstance(cls, types.UnionType)
            and len(cls.__args__) == 2
            and cls.__args__[1] == type(None)
            and is_dataclass(cls.__args__[0])
            and isinstance(cfg, dict)
        ):
            fieldtypes = typing.get_type_hints(cls.__args__[0])
            return cls.__args__[0](
                **{k: __dict_to_dataclass(v, fieldtypes[k]) for k, v in cfg.items()}
            )
        if typing.get_origin(cls) == list:
            return [__dict_to_dataclass(v, typing.get_args(cls)[0]) for v in cfg]
        if cls == Path or (
            isinstance(cls, types.UnionType)
            and Path in cls.__args__
            and cfg is not None
        ):
            return Path(cfg)
        return cfg

    return __dict_to_dataclass(cfg, Config)  # type: ignore


def omegaconf_resolve(config: Config):
    """Resolve all variable interpolations in config object with OmegaConf

    Args:
        config: Config object to resolve

    Returns:
        resolved config object
    """
    dict_config = asdict(config)

    # convert all paths to string, omegaconf does not do variable interpolation in anything that's not a string
    dict_config = _path_to_str(dict_config)

    # omegaconf can't handle callables, may need to extend this list if other callable configs get added
    del dict_config["trainer"]["lr_scheduler"]
    del dict_config["trainer"]["optimizer"]

    oc_config = OmegaConf.create(dict_config)
    dict_config: dict[str, Any] = OmegaConf.to_object(oc_config)  # type: ignore

    dict_config["trainer"]["lr_scheduler"] = config.trainer.lr_scheduler
    dict_config["trainer"]["optimizer"] = config.trainer.optimizer

    new_config = _dict_to_dataclass(dict_config)
    return new_config


def setup_logging():
    from fd_shifts import logger

    rich.reconfigure(stderr=True, force_terminal=True)
    logger.remove()  # Remove default 'stderr' handler

    # We need to specify end=''" as log message already ends with \n (thus the lambda function)
    # Also forcing 'colorize=True' otherwise Loguru won't recognize that the sink support colors
    logger.add(
        lambda m: rich.get_console().print(m, end="", markup=False, highlight=False),
        colorize=True,
        enqueue=True,
        level="DEBUG",
        backtrace=True,
        diagnose=True,
    )


@subcommand
def train(config: Config):
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar
    from pytorch_lightning.loggers.csv_logs import CSVLogger
    from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
    from pytorch_lightning.loggers.wandb import WandbLogger

    from fd_shifts import logger
    from fd_shifts.loaders.data_loader import FDShiftsDataLoader
    from fd_shifts.models import get_model
    from fd_shifts.models.callbacks import get_callbacks
    from fd_shifts.utils import exp_utils

    progress = RichProgressBar(console_kwargs={"stderr": True, "force_terminal": True})

    if config.exp.dir is None:
        raise ValueError("Experiment directory must be specified")
    config.exp.version = exp_utils.get_next_version(config.exp.dir)
    # HACK: This should be automatically linked or not configurable
    config.exp.version_dir = config.exp.dir / f"version_{config.exp.version}"

    logger.info(pretty_repr(config))

    # TODO: Clean the rest of this up

    max_steps = (
        config.trainer.num_steps if hasattr(config.trainer, "num_steps") else None
    )
    accumulate_grad_batches = (
        config.trainer.accumulate_grad_batches
        if hasattr(config.trainer, "accumulate_grad_batches")
        else 1
    )

    limit_batches: float | int = 1.0
    num_epochs = config.trainer.num_epochs
    val_every_n_epoch = config.trainer.val_every_n_epoch
    log_every_n_steps = 50

    if isinstance(config.trainer.fast_dev_run, bool):
        limit_batches = 1 if config.trainer.fast_dev_run else 1.0
        num_epochs = 1 if config.trainer.fast_dev_run else num_epochs
        max_steps = 1 if config.trainer.fast_dev_run else max_steps
        val_every_n_epoch = 1 if config.trainer.fast_dev_run else val_every_n_epoch
    elif isinstance(config.trainer.fast_dev_run, int):
        limit_batches = config.trainer.fast_dev_run * accumulate_grad_batches
        max_steps = config.trainer.fast_dev_run * 2
        config.trainer.dg_pretrain_epochs = None
        config.trainer.dg_pretrain_steps = (max_steps * 2) // 3
        val_every_n_epoch = 1
        log_every_n_steps = 1
        num_epochs = None

    datamodule = FDShiftsDataLoader(config)
    model = get_model(config.model.name)(config)
    csv_logger = CSVLogger(
        save_dir=str(config.exp.group_dir),
        name=config.exp.name,
        version=config.exp.version,
    )

    tb_logger = TensorBoardLogger(
        save_dir=str(config.exp.group_dir),
        name=config.exp.name,
        default_hp_metric=False,
    )

    wandb_logger = WandbLogger(
        project="fd_shifts_proto",
        name=config.exp.name,
    )

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        logger=[tb_logger, csv_logger, wandb_logger],
        log_every_n_steps=log_every_n_steps,
        max_epochs=num_epochs,
        max_steps=max_steps,  # type: ignore
        callbacks=[progress] + get_callbacks(config),
        benchmark=config.trainer.benchmark,
        precision=16,
        check_val_every_n_epoch=val_every_n_epoch,
        num_sanity_val_steps=5,
        limit_train_batches=limit_batches,
        limit_val_batches=0 if config.trainer.do_val is False else limit_batches,
        limit_test_batches=limit_batches,
        gradient_clip_val=1,
        accumulate_grad_batches=accumulate_grad_batches,
    )

    logger.info(
        "logging training to: {}, version: {}".format(
            config.exp.dir, config.exp.version
        )
    )
    trainer.fit(model=model, datamodule=datamodule)


@subcommand
def test(config: Config):
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar
    from pytorch_lightning.loggers.wandb import WandbLogger

    from fd_shifts import logger
    from fd_shifts.loaders.data_loader import FDShiftsDataLoader
    from fd_shifts.models import get_model
    from fd_shifts.models.callbacks import get_callbacks
    from fd_shifts.utils import exp_utils

    progress = RichProgressBar(console_kwargs={"stderr": True, "force_terminal": True})

    if config.exp.dir is None:
        raise ValueError("Experiment directory must be specified")

    config.exp.version = (
        version if (version := exp_utils.get_most_recent_version(config.exp.dir)) else 0
    )
    # HACK: This should be automatically linked or not configurable
    config.exp.version_dir = config.exp.dir / f"version_{config.exp.version}"
    logger.info(pretty_repr(config))

    ckpt_path = exp_utils._get_resume_ckpt_path(config)

    logger.info(
        "testing model from checkpoint: {} from model selection tpye {}".format(
            ckpt_path, config.test.selection_criterion
        )
    )
    logger.info("logging testing to: {}".format(config.test.dir))

    module = get_model(config.model.name)(config)

    # TODO: make common module class with this method
    module.load_only_state_dict(ckpt_path)  # type: ignore

    datamodule = FDShiftsDataLoader(config)

    if not config.test.dir.exists():
        config.test.dir.mkdir(parents=True)

    limit_batches: float | int = 1.0
    log_every_n_steps = 50

    if isinstance(config.trainer.fast_dev_run, bool):
        limit_batches = 1 if config.trainer.fast_dev_run else 1.0
    elif isinstance(config.trainer.fast_dev_run, int):
        limit_batches = config.trainer.fast_dev_run
        log_every_n_steps = 1

    wandb_logger = WandbLogger(
        project="fd_shifts_proto",
        name=config.exp.name,
    )

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        logger=wandb_logger,
        log_every_n_steps=log_every_n_steps,
        callbacks=[progress] + get_callbacks(config),
        limit_test_batches=limit_batches,
        precision=16,
    )
    trainer.test(model=module, datamodule=datamodule)


@subcommand
def analysis(config: Config):
    from fd_shifts import analysis as ana

    ana.main(
        in_path=config.test.dir,
        out_path=config.exp.output_paths.analysis,
        query_studies=config.eval.query_studies,
        add_val_tuning=config.eval.val_tuning,
        threshold_plot_confid=None,
        cf=config,
    )


@subcommand
def debug(config: Config):
    pass


def _list_experiments():
    from fd_shifts.experiments.configs import list_experiment_configs

    rich.print("Available experiments:")
    for exp in sorted(list_experiment_configs()):
        rich.print(exp)


def get_parser():
    from fd_shifts import get_version

    parser = ArgumentParser(version=get_version())
    parser.add_argument("-f", "--overwrite-config-file", action="store_true")
    subcommands = parser.add_subcommands(dest="command")
    subparsers: dict[str, ArgumentParser] = {}

    subparser = ArgumentParser()
    subcommands.add_subcommand("list-experiments", subparser)

    for name, func in __subcommands.items():
        subparser = ArgumentParser()
        subparser.add_argument(
            "--config-file", "--legacy-config-file", action=ActionLegacyConfigFile
        )
        subparser.add_argument("--experiment", action=ActionExperiment)
        subparser.add_function_arguments(func, sub_configs=True)
        subparsers[name] = subparser
        subcommands.add_subcommand(name, subparser)

    return parser, subparsers


def config_from_parser(parser, args):
    config = parser.instantiate_classes(args)[args.command].config
    config = omegaconf_resolve(config)
    return config


def main():
    setup_logging()

    parser, subparsers = get_parser()

    args = parser.parse_args()

    if args.command == "list-experiments":
        _list_experiments()
        return

    config = config_from_parser(parser, args)

    rich.print(config)

    # TODO: Check if configs are the same
    config.test.cf_path.parent.mkdir(parents=True, exist_ok=True)
    subparsers[args.command].save(
        args[args.command],
        config.test.cf_path,
        skip_check=True,
        overwrite=args.overwrite_config_file,
    )

    __subcommands[args.command](config=config)


if __name__ == "__main__":
    main()
