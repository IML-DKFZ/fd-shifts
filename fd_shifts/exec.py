import os
import random
from typing import cast

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBar
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from rich import get_console, reconfigure
from torch import multiprocessing

from fd_shifts import analysis, configs, logger
from fd_shifts.loaders.data_loader import FDShiftsDataLoader
from fd_shifts.models import get_model
from fd_shifts.models.callbacks import get_callbacks
from fd_shifts.utils import exp_utils

configs.init()


def train(
    cf: configs.Config,
    progress: RichProgressBar = RichProgressBar(),
    subsequent_testing: bool = False,
) -> None:
    """
    perform the training routine for a given fold. saves plots and selected parameters to the experiment dir
    specified in the configs.
    """

    logger.info("CHECK CUDNN VERSION", torch.backends.cudnn.version())
    train_deterministic_flag = False
    if cf.exp.global_seed is not None:
        exp_utils.set_seed(cf.exp.global_seed)
        cf.trainer.benchmark = False
        logger.info(
            "setting seed {}, benchmark to False for deterministic training.".format(
                cf.exp.global_seed
            )
        )

    resume_ckpt_path = None
    cf.exp.version = exp_utils.get_next_version(cf.exp.dir)
    cf.exp.version_dir = cf.exp.version_dir.parent / f"version_{cf.exp.version}"
    if cf.trainer.resume_from_ckpt:
        cf.exp.version -= 1
        resume_ckpt_path = exp_utils._get_resume_ckpt_path(cf)
        logger.info("resuming previous training:", resume_ckpt_path)

    if cf.trainer.resume_from_ckpt_confidnet:
        cf.exp.version -= 1
        cf.trainer.callbacks.training_stages.pretrained_confidnet_path = (
            exp_utils._get_resume_ckpt_path(cf)
        )
        logger.info("resuming previous training:", resume_ckpt_path)

    if "openset" in cf.data.dataset:
        cf.data.kwargs["out_classes"] = cf.data.kwargs.get(
            "out_classes",
            random.sample(range(cf.data.num_classes), int(0.4 * cf.data.num_classes)),
        )

    max_steps = cf.trainer.num_steps if hasattr(cf.trainer, "num_steps") else None
    accumulate_grad_batches = (
        cf.trainer.accumulate_grad_batches
        if hasattr(cf.trainer, "accumulate_grad_batches")
        else 1
    )

    limit_batches: float | int = 1.0
    num_epochs = cf.trainer.num_epochs
    val_every_n_epoch = cf.trainer.val_every_n_epoch

    if isinstance(cf.trainer.fast_dev_run, bool):
        limit_batches = 1 if cf.trainer.fast_dev_run else 1.0
        num_epochs = 1 if cf.trainer.fast_dev_run else num_epochs
        max_steps = 1 if cf.trainer.fast_dev_run else max_steps
        val_every_n_epoch = 1 if cf.trainer.fast_dev_run else val_every_n_epoch
    elif isinstance(cf.trainer.fast_dev_run, int):
        limit_batches = cf.trainer.fast_dev_run * accumulate_grad_batches
        max_steps = cf.trainer.fast_dev_run * 5
        cf.trainer.dg_pretrain_epochs = None
        cf.trainer.dg_pretrain_steps = (max_steps * 2) // 3
        val_every_n_epoch = 1
        num_epochs = None

    datamodule = FDShiftsDataLoader(cf)
    model = get_model(cf.model.name)(cf)
    tb_logger = TensorBoardLogger(
        save_dir=str(cf.exp.group_dir),
        name=cf.exp.name,
        default_hp_metric=False,
    )
    csv_logger = CSVLogger(
        save_dir=str(cf.exp.group_dir), name=cf.exp.name, version=cf.exp.version
    )

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        logger=[tb_logger, csv_logger],
        max_epochs=num_epochs,
        max_steps=max_steps,
        callbacks=[progress] + get_callbacks(cf),
        resume_from_checkpoint=resume_ckpt_path,
        benchmark=cf.trainer.benchmark,
        check_val_every_n_epoch=val_every_n_epoch,
        num_sanity_val_steps=5,
        deterministic=train_deterministic_flag,
        limit_train_batches=limit_batches,
        limit_val_batches=0 if cf.trainer.do_val is False else limit_batches,
        limit_test_batches=limit_batches,
        gradient_clip_val=1,
        accumulate_grad_batches=accumulate_grad_batches,
    )

    logger.info(
        "logging training to: {}, version: {}".format(cf.exp.dir, cf.exp.version)
    )
    trainer.fit(model=model, datamodule=datamodule)

    if subsequent_testing:
        test(cf, progress)


def test(cf: configs.Config, progress: RichProgressBar = RichProgressBar()) -> None:
    """Run inference

    Args:
        cf (configs.Config): configuration object to run inference on
        progress: (RichProgressBar): global progress bar
    """
    if "best" in cf.test.selection_criterion and cf.test.only_latest_version is False:
        ckpt_path = exp_utils._get_path_to_best_ckpt(
            cf.exp.dir, cf.test.selection_criterion, cf.test.selection_mode
        )
    else:
        logger.info("CHECK cf.exp.dir", cf.exp.dir)
        cf.exp.version = exp_utils.get_most_recent_version(cf.exp.dir)
        cf.exp.version_dir = cf.exp.version_dir.parent / f"version_{cf.exp.version}"
        ckpt_path = exp_utils._get_resume_ckpt_path(cf)

    logger.info(
        "testing model from checkpoint: {} from model selection tpye {}".format(
            ckpt_path, cf.test.selection_criterion
        )
    )
    logger.info("logging testing to: {}".format(cf.test.dir))

    module = get_model(cf.model.name)(cf)
    module.load_only_state_dict(ckpt_path)
    datamodule = FDShiftsDataLoader(cf)

    if not os.path.exists(cf.test.dir):
        os.makedirs(cf.test.dir)

    limit_batches: float | int = 1.0

    if isinstance(cf.trainer.fast_dev_run, bool):
        limit_batches = 1 if cf.trainer.fast_dev_run else 1.0
    elif isinstance(cf.trainer.fast_dev_run, int):
        limit_batches = cf.trainer.fast_dev_run

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        logger=False,
        callbacks=[progress] + get_callbacks(cf),
        limit_test_batches=limit_batches,
        replace_sampler_ddp=False,
    )
    trainer.test(model=module, datamodule=datamodule)
    analysis.main(
        in_path=cf.test.dir,
        out_path=cf.test.dir,
        query_studies=cf.eval.query_studies,
        add_val_tuning=cf.eval.val_tuning,
        threshold_plot_confid=None,
        cf=cf,
    )


@hydra.main(config_path="configs", config_name="config", version_base="1.1")
def main(dconf: DictConfig) -> None:
    """main entry point for running anything with a Trainer

    Args:
        dconf (DictConfig): config passed in by hydra
    """
    multiprocessing.set_start_method("spawn")

    reconfigure(stderr=True, force_terminal=True)
    progress = RichProgressBar(console_kwargs={"stderr": True, "force_terminal": True})
    logger.remove()  # Remove default 'stderr' handler

    # We need to specify end=''" as log message already ends with \n (thus the lambda function)
    # Also forcing 'colorize=True' otherwise Loguru won't recognize that the sink support colors
    logger.add(
        lambda m: get_console().print(m, end="", markup=False, highlight=False),
        colorize=True,
        enqueue=True,
        level="DEBUG",
        backtrace=True,
        diagnose=True,
    )

    try:
        # NOTE: Needed because hydra does not set this if we load a previous experiment
        dconf._metadata.object_type = configs.Config

        def _fix_metadata(cfg: DictConfig) -> None:
            if hasattr(cfg, "_target_"):
                cfg._metadata.object_type = getattr(
                    configs, cfg._target_.split(".")[-1]
                )
            for _, v in cfg.items():
                match v:
                    case DictConfig():  # type: ignore
                        _fix_metadata(v)
                    case _:
                        pass

        _fix_metadata(dconf)
        conf: configs.Config = cast(configs.Config, OmegaConf.to_object(dconf))

        conf.__pydantic_validate_values__()

        if conf.exp.mode == configs.Mode.train:
            conf.exp.version = exp_utils.get_next_version(conf.exp.dir)
            if conf.trainer.resume_from_ckpt:
                conf.exp.version -= 1

            if conf.trainer.resume_from_ckpt_confidnet:
                conf.exp.version -= 1
            conf.data.num_workers = exp_utils._get_allowed_n_proc_DA(
                conf.data.num_workers
            )

            conf.__pydantic_validate_values__()
            logger.info(OmegaConf.to_yaml(conf))

            train(conf, progress)

        elif conf.exp.mode == configs.Mode.train_test:
            conf.exp.version = exp_utils.get_next_version(conf.exp.dir)
            if conf.trainer.resume_from_ckpt:
                conf.exp.version -= 1

            if conf.trainer.resume_from_ckpt_confidnet:
                conf.exp.version -= 1
            conf.data.num_workers = exp_utils._get_allowed_n_proc_DA(
                conf.data.num_workers
            )

            conf.__pydantic_validate_values__()
            logger.info(OmegaConf.to_yaml(conf))
            train(conf, progress, subsequent_testing=True)

        elif conf.exp.mode == configs.Mode.test:
            if (
                "best" in conf.test.selection_criterion
                and conf.test.only_latest_version is False
            ):
                ckpt_path = exp_utils._get_path_to_best_ckpt(
                    conf.exp.dir,
                    conf.test.selection_criterion,
                    conf.test.selection_mode,
                )
            else:
                logger.info("CHECK conf.exp.dir", conf.exp.dir)
                conf.exp.version = exp_utils.get_most_recent_version(conf.exp.dir)
                ckpt_path = exp_utils._get_resume_ckpt_path(conf)
            conf.__pydantic_validate_values__()
            logger.info(OmegaConf.to_yaml(conf))
            test(conf, progress)

        elif conf.exp.mode == configs.Mode.analysis:
            conf.__pydantic_validate_values__()
            logger.info(OmegaConf.to_yaml(conf))
            analysis.main(
                in_path=conf.test.dir,
                out_path=conf.test.dir,
                query_studies=conf.eval.query_studies,
                add_val_tuning=conf.eval.val_tuning,
                threshold_plot_confid=None,
                cf=conf,
            )
        else:
            conf.__pydantic_validate_values__()
            logger.info("BEGIN CONFIG\n{}\nEND CONFIG", OmegaConf.to_yaml(conf))
    except Exception as e:
        logger.exception(e)
        raise e


if __name__ == "__main__":
    main()
