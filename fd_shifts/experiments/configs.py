import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pl_bolts
import torch
from omegaconf import SI

from fd_shifts.configs import (
    ConfidMeasuresConfig,
    ConfidMetricsConfig,
    Config,
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    LRSchedulerConfig,
    Mode,
    ModelConfig,
    NetworkConfig,
    OptimizerConfig,
    OutputPathsConfig,
    OutputPathsPerMode,
    PerfMetricsConfig,
    QueryStudiesConfig,
    TestConfig,
    TrainerConfig,
    ValSplit,
)

__data_configs = {}

__data_configs["svhn_384"] = DataConfig(
    dataset="svhn",
    data_dir=SI("${oc.env:DATASET_ROOT_DIR}/svhn"),
    pin_memory=True,
    img_size=(384, 384, 3),
    num_workers=24,
    num_classes=10,
    reproduce_confidnet_splits=True,
    augmentations={
        "train": {
            "to_tensor": None,
            "resize": 384,
            "normalize": [
                [0.4376821, 0.4437697, 0.47280442],
                [0.19803012, 0.20101562, 0.19703614],
            ],
        },
        "val": {
            "to_tensor": None,
            "resize": 384,
            "normalize": [
                [0.4376821, 0.4437697, 0.47280442],
                [0.19803012, 0.20101562, 0.19703614],
            ],
        },
        "test": {
            "to_tensor": None,
            "resize": 384,
            "normalize": [
                [0.4376821, 0.4437697, 0.47280442],
                [0.19803012, 0.20101562, 0.19703614],
            ],
        },
    },
    target_transforms=None,
    kwargs=None,
)


def get_data_config(name: str) -> DataConfig:
    return __data_configs[name]


__experiments = {}

__experiments["svhn_modeldg_bbvit_lr0.01_bs128_run4_do1_rew10"] = Config(
    data=get_data_config("svhn_384"),
    trainer=TrainerConfig(
        val_every_n_epoch=5,
        do_val=True,
        batch_size=128,
        resume_from_ckpt=False,
        benchmark=True,
        fast_dev_run=False,
        lr_scheduler=LRSchedulerConfig(
            {
                "class_path": "pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR",
                "init_args": {
                    "warmup_epochs": 500,
                    "max_epochs": 60000,
                    "warmup_start_lr": 0.0,
                    "eta_min": 0.0,
                    "last_epoch": -1,
                },
            }
        ),
        optimizer=OptimizerConfig(
            {
                "class_path": "torch.optim.SGD",
                "init_args": {
                    "lr": 0.01,
                    "dampening": 0.0,
                    "momentum": 0.9,
                    "nesterov": False,
                    "maximize": False,
                    "weight_decay": 0.0,
                },
            }
        ),
        accumulate_grad_batches=1,
        resume_from_ckpt_confidnet=False,
        num_epochs=None,
        num_steps=60000,
        num_epochs_backbone=None,
        dg_pretrain_epochs=None,
        dg_pretrain_steps=20000,
        val_split=ValSplit.devries,
        lr_scheduler_interval="step",
        callbacks={
            "model_checkpoint": None,
            "confid_monitor": None,
            "learning_rate_monitor": None,
        },
        learning_rate_confidnet=None,
        learning_rate_confidnet_finetune=None,
    ),
    exp=ExperimentConfig(
        group_name="vit",
        name="svhn_modeldg_bbvit_lr0.01_bs128_run4_do1_rew10",
        mode=Mode.analysis,
        work_dir=Path.cwd(),
        fold_dir=SI("exp/${exp.fold}"),
        root_dir=Path(p)
        if (p := os.getenv("EXPERIMENT_ROOT_DIR")) is not None
        else None,
        data_root_dir=Path(p)
        if (p := os.getenv("DATASET_ROOT_DIR")) is not None
        else None,
        group_dir=Path("${exp.root_dir}/${exp.group_name}"),
        dir=Path("${exp.group_dir}/${exp.name}"),
        version_dir=Path("${exp.dir}/version_${exp.version}"),
        fold=0,
        crossval_n_folds=10,
        crossval_ids_path=Path("${exp.dir}/crossval_ids.pickle"),
        log_path=Path("log.txt"),
        global_seed=0,
        output_paths=OutputPathsPerMode(
            fit=OutputPathsConfig(
                raw_output=Path("${exp.version_dir}/raw_output.npz"),
                raw_output_dist=Path("${exp.version_dir}/raw_output_dist.npz"),
                external_confids=Path("${exp.version_dir}/external_confids.npz"),
                external_confids_dist=Path(
                    "${exp.version_dir}/external_confids_dist.npz"
                ),
                input_imgs_plot=Path("${exp.dir}/input_imgs.png"),
                encoded_output=None,
                attributions_output=None,
            ),
            test=OutputPathsConfig(
                raw_output=Path("${test.dir}/raw_logits.npz"),
                raw_output_dist=Path("${test.dir}/raw_logits_dist.npz"),
                external_confids=Path("${test.dir}/external_confids.npz"),
                external_confids_dist=Path("${test.dir}/external_confids_dist.npz"),
                input_imgs_plot=None,
                encoded_output=Path("${test.dir}/encoded_output.npz"),
                attributions_output=Path("${test.dir}/attributions.csv"),
            ),
        ),
        version=None,
    ),
    model=ModelConfig(
        name="devries_model",
        network=NetworkConfig(
            name="vit",
            backbone=None,
            imagenet_weights_path=None,
            load_dg_backbone_path=None,
            save_dg_backbone_path=Path("${exp.dir}/dg_backbone.ckpt"),
        ),
        fc_dim=768,
        avg_pool=True,
        dropout_rate=1,
        monitor_mcd_samples=50,
        test_mcd_samples=50,
        confidnet_fc_dim=None,
        dg_reward=10,
        balanced_sampeling=False,
        budget=0.3,
    ),
    eval=EvalConfig(
        tb_hparams=["fold"],
        test_conf_scaling=False,
        val_tuning=True,
        r_star=0.25,
        r_delta=0.05,
        query_studies=QueryStudiesConfig(
            iid_study="svhn_384",
            noise_study=[],
            in_class_study=[],
            new_class_study=["cifar10_384", "cifar100_384", "tinyimagenet_384"],
        ),
        performance_metrics=PerfMetricsConfig(
            train=["loss", "nll", "accuracy"],
            val=["loss", "nll", "accuracy", "brier_score"],
            test=["nll", "accuracy", "brier_score"],
        ),
        confid_metrics=ConfidMetricsConfig(
            train=[
                "failauc",
                "failap_suc",
                "failap_err",
                "fpr@95tpr",
                "e-aurc",
                "aurc",
            ],
            val=["failauc", "failap_suc", "failap_err", "fpr@95tpr", "e-aurc", "aurc"],
            test=[
                "failauc",
                "failap_suc",
                "failap_err",
                "mce",
                "ece",
                "b-aurc",
                "e-aurc",
                "aurc",
                "fpr@95tpr",
            ],
        ),
        confidence_measures=ConfidMeasuresConfig(
            train=["det_mcp"], val=["det_mcp"], test=["det_mcp", "det_pe", "ext"]
        ),
        monitor_plots=["hist_per_confid"],
        ext_confid_name="dg",
    ),
    test=TestConfig(
        name="test_results",
        dir=Path("${exp.dir}/${test.name}"),
        cf_path=Path("${exp.dir}/hydra/config.yaml"),
        selection_criterion="latest",
        best_ckpt_path=Path("${exp.version_dir}/${test.selection_criterion}.ckpt"),
        only_latest_version=True,
        devries_repro_ood_split=False,
        assim_ood_norm_flag=False,
        iid_set_split="devries",
        raw_output_path="raw_output.npz",
        external_confids_output_path="external_confids.npz",
        output_precision=16,
        selection_mode="max",
    ),
)


def get_experiment_config(name: str) -> Config:
    return __experiments[name]
