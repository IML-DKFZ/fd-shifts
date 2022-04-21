import logging
import os
from pathlib import Path
import random
import sys

import hydra
import pytorch_lightning as pl
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from rich.console import Console
from torch import multiprocessing

from fd_shifts import analysis
from fd_shifts.loaders.abstract_loader import AbstractDataLoader
from fd_shifts.models import get_model
from fd_shifts.models.callbacks import get_callbacks
from fd_shifts.utils import exp_utils


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


def train(cf, subsequent_testing=False):
    """
    perform the training routine for a given fold. saves plots and selected parameters to the experiment dir
    specified in the configs.
    """

    console = Console(stderr=True, force_terminal=True)
    progress = RichProgressBar(console_kwargs={"stderr": True, "force_terminal": True})
    progress._console = console
    logger.remove()  # Remove default 'stderr' handler

    # We need to specify end=''" as log message already ends with \n (thus the lambda function)
    # Also forcing 'colorize=True' otherwise Loguru won't recognize that the sink support colors
    logger.add(
        lambda m: progress._console.print(m, end="", markup=False, highlight=False),
        colorize=True,
        enqueue=True,
        level="DEBUG",
    )

    logger.info("CHECK CUDNN VERSION", torch.backends.cudnn.version())
    train_deterministic_flag = False
    if cf.exp.global_seed is not False:
        # exp_utils.set_seed(cf.exp.global_seed)
        exp_utils.set_seed(cf.exp.global_seed)
        cf.trainer.benchmark = False
        train_deterministic_flag = True
        logger.info(
            "setting seed {}, benchmark to False for deterministic training.".format(
                cf.exp.global_seed
            )
        )

    resume_ckpt_path = None
    cf.exp.version = exp_utils.get_next_version(cf.exp.dir)
    if cf.trainer.resume_from_ckpt:
        cf.exp.version -= 1
        resume_ckpt_path = exp_utils.get_resume_ckpt_path(cf)
        logger.info("resuming previous training:", resume_ckpt_path)

    if cf.trainer.resume_from_ckpt_confidnet:
        cf.exp.version -= 1
        cf.trainer.callbacks.training_stages.pretrained_confidnet_path = (
            exp_utils.get_resume_ckpt_path(cf)
        )
        logger.info("resuming previous training:", resume_ckpt_path)

    # TODO: Don't hard-code number of total classes and number of holdout classes
    if "openset" in cf.data.dataset:
        cf.data.kwargs.out_classes = random.sample(
            range(cf.data.num_classes), int(0.4 * cf.data.num_classes)
        )

    datamodule = AbstractDataLoader(cf)
    model = get_model(cf.model.name)(cf)
    tb_logger = TensorBoardLogger(
        save_dir=cf.exp.group_dir,
        name=cf.exp.name,
        default_hp_metric=False,
    )
    cf.exp.version = tb_logger.version
    csv_logger = CSVLogger(
        save_dir=cf.exp.group_dir, name=cf.exp.name, version=cf.exp.version
    )

    max_steps = cf.trainer.num_steps if hasattr(cf.trainer, "num_steps") else None
    accelerator = cf.trainer.accelerator if hasattr(cf.trainer, "accelerator") else None
    accumulate_grad_batches = (
        cf.trainer.accumulate_grad_batches
        if hasattr(cf.trainer, "accumulate_grad_batches")
        else 1
    )

    trainer = pl.Trainer(
        gpus=-1,
        logger=[tb_logger, csv_logger],
        max_epochs=cf.trainer.num_epochs,
        max_steps=max_steps,
        callbacks=[progress] + get_callbacks(cf),
        resume_from_checkpoint=resume_ckpt_path,
        benchmark=cf.trainer.benchmark,
        check_val_every_n_epoch=cf.trainer.val_every_n_epoch,
        fast_dev_run=cf.trainer.fast_dev_run,
        num_sanity_val_steps=5,
        # amp_level="O0",
        deterministic=train_deterministic_flag,
        # limit_train_batches=50,
        limit_val_batches=0 if cf.trainer.do_val is False else 1.0,
        # replace_sampler_ddp=False,
        accelerator=accelerator,
        gradient_clip_val=1,
        accumulate_grad_batches=accumulate_grad_batches,
    )

    logger.info(
        "logging training to: {}, version: {}".format(cf.exp.dir, cf.exp.version)
    )
    trainer.fit(model=model, datamodule=datamodule)
    # analysis.main(in_path=cf.exp.version_dir,
    #               out_path=cf.exp.version_dir,
    #               query_studies={"iid_study": cf.data.dataset})

    if subsequent_testing:

        if not os.path.exists(cf.test.dir):
            os.makedirs(cf.test.dir)

        if cf.test.selection_criterion == "latest":
            ckpt_path = None
            logger.info("testing with latest model...")
        elif "best" in cf.test.selection_criterion:
            ckpt_path = cf.test.best_ckpt_path
            logger.info(
                "testing with best model from {} and epoch {}".format(
                    cf.test.best_ckpt_path, torch.load(ckpt_path)["epoch"]
                )
            )
        trainer.test(ckpt_path=ckpt_path)
        analysis.main(
            in_path=cf.test.dir,
            out_path=cf.test.dir,
            query_studies=cf.eval.query_studies,
            add_val_tuning=cf.eval.val_tuning,
            cf=cf,
        )


def test(cf):

    console = Console(stderr=True, force_terminal=True)
    progress = RichProgressBar(console_kwargs={"stderr": True, "force_terminal": True})
    progress._console = console
    logger.remove()  # Remove default 'stderr' handler

    # We need to specify end=''" as log message already ends with \n (thus the lambda function)
    # Also forcing 'colorize=True' otherwise Loguru won't recognize that the sink support colors
    logger.add(
        lambda m: progress._console.print(m, end="", markup=False, highlight=False),
        colorize=True,
        enqueue=True,
        level="DEBUG",
    )

    if "best" in cf.test.selection_criterion and cf.test.only_latest_version is False:
        ckpt_path = exp_utils.get_path_to_best_ckpt(
            cf.exp.dir, cf.test.selection_criterion, cf.test.selection_mode
        )
    else:
        logger.info("CHECK cf.exp.dir", cf.exp.dir)
        cf.exp.version = exp_utils.get_most_recent_version(cf.exp.dir)
        ckpt_path = exp_utils.get_resume_ckpt_path(cf)

    logger.info(
        "testing model from checkpoint: {} from model selection tpye {}".format(
            ckpt_path, cf.test.selection_criterion
        )
    )
    logger.info("logging testing to: {}".format(cf.test.dir))

    module = get_model(cf.model.name)(cf)
    module.load_only_state_dict(ckpt_path)
    datamodule = AbstractDataLoader(cf)

    if not os.path.exists(cf.test.dir):
        os.makedirs(cf.test.dir)

    accelerator = cf.trainer.accelerator if hasattr(cf.trainer, "accelerator") else None
    trainer = pl.Trainer(
        gpus=-1,
        logger=False,
        callbacks=[progress] + get_callbacks(cf),
        precision=64,
        replace_sampler_ddp=False,
        # accelerator="ddp",
        accelerator=None,
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

    # fix str bug
    # test resuming by testing a second time in the same dir
    # how to print the tested epoch into csv log?


@hydra.main(config_path="configs", config_name="config")
def main(cf: DictConfig):
    multiprocessing.set_start_method("spawn")

    sys.stdout = exp_utils.Logger(cf.exp.log_path)
    sys.stderr = exp_utils.Logger(cf.exp.log_path)
    logger.info(OmegaConf.to_yaml(cf))
    cf.data.num_workers = exp_utils.get_allowed_n_proc_DA(cf.data.num_workers)

    if cf.exp.mode == "train":
        train(cf)

    if cf.exp.mode == "train_test":
        train(cf, subsequent_testing=True)

    if cf.exp.mode == "test":
        test(cf)

    if cf.exp.mode == "analysis":
        analysis.main(
            in_path=cf.test.dir,
            out_path=cf.test.dir,
            query_studies=cf.eval.query_studies,
            add_val_tuning=cf.eval.val_tuning,
            threshold_plot_confid=None,
            cf=cf,
        )


if __name__ == "__main__":
    main()
