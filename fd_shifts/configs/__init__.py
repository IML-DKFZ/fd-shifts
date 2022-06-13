from dataclasses import field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional, cast

import hydra
import pl_bolts
import torch
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra_zen import ZenField, builds
from omegaconf import DictConfig
from omegaconf.omegaconf import MISSING
from pydantic import validator
from pydantic.dataclasses import dataclass

from fd_shifts import models
from fd_shifts.analysis import confid_scores, metrics
from fd_shifts.loaders import dataset_collection

from ..models import networks

# TODO: Clean up data configs (-> instantiation? enum?)
# TODO: Clean up model configs (-> instantiation? enum?)


class Mode(Enum):
    train = auto()
    test = auto()
    train_test = auto()


class ValSplit(Enum):
    devries = auto()
    repro_confidnet = auto()
    cv = auto()
    zhang = auto()  # TODO: Should this still be here?


@dataclass
class OutputPathsConfig:
    input_imgs_plot: Optional[Path] = Path("${exp.dir}/input_imgs.png")
    raw_output: Path = Path("${exp.version_dir}/raw_output.npz")
    raw_output_dist: Path = Path("${exp.version_dir}/raw_output_dist.npz")
    external_confids: Path = Path("${exp.version_dir}/external_confids.npz")
    external_confids_dist: Path = Path("${exp.version_dir}/external_confids_dist.npz")


@dataclass
class OutputPathsPerMode:
    fit: OutputPathsConfig = OutputPathsConfig()
    test: OutputPathsConfig = OutputPathsConfig(
        input_imgs_plot=None,
        raw_output=Path("${test.dir}/raw_output.npz"),
        raw_output_dist=Path("${test.dir}/raw_output_dist.npz"),
        external_confids=Path("${test.dir}/external_confids.npz"),
        external_confids_dist=Path("${test.dir}/external_confids_dist.npz"),
    )


@dataclass
class ExperimentConfig:
    group_name: str = MISSING
    name: str = MISSING
    version: Optional[int] = None
    mode: Mode = Mode.train_test  # train or test
    work_dir: Path = Path("${hydra:runtime.cwd}")
    fold_dir: Path = Path("exp/${exp.fold}")
    root_dir: Path = Path("${env:EXPERIMENT_ROOT_DIR}")
    data_root_dir: Path = Path("${env:DATASET_ROOT_DIR}")
    group_dir: Path = Path("${env:EXPERIMENT_ROOT_DIR}/${exp.group_name}")
    dir: Path = Path("${exp.group_dir}/${exp.name}")
    version_dir: Path = Path("${exp.dir}/version_${exp.version}")
    fold: int = 0
    crossval_n_folds: int = 10
    crossval_ids_path: Path = Path("${exp.dir}/crossval_ids.pickle")
    output_paths: OutputPathsPerMode = OutputPathsPerMode()
    log_path: Path = Path("./log.txt")
    global_seed: Optional[int] = None


@dataclass
class LRSchedulerConfig:
    _target_: str = MISSING


CosineAnnealingLR = builds(
    torch.optim.lr_scheduler.CosineAnnealingLR,
    builds_bases=(LRSchedulerConfig,),
    zen_partial=True,
    populate_full_signature=True,
    T_max="${trainer.num_steps}",
)

LinearWarmupCosineAnnealingLR = builds(
    pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR,
    builds_bases=(LRSchedulerConfig,),
    zen_partial=True,
    populate_full_signature=True,
    max_epochs="${trainer.num_steps}",
    warmup_epochs=500,
)


@dataclass
class OptimizerConfig:
    _target_: str = MISSING


SGD = builds(
    torch.optim.SGD,
    lr=0.003,
    momentum=0.9,
    weight_decay=0.0,
    builds_bases=(OptimizerConfig,),
    zen_partial=True,
    populate_full_signature=True,
)


@dataclass
class TrainerConfig:
    resume_from_ckpt_confidnet: bool = False
    num_epochs: Optional[
        int
    ] = 300  # 250 has to be >1 because of incompatibility of lighting eval with psuedo test
    num_steps: Optional[
        int
    ] = 300  # 250 has to be >1 because of incompatibility of lighting eval with psuedo test
    num_epochs_backbone: Optional[int] = None
    dg_pretrain_epochs: int = 100  # 100 and 300 total epochs
    val_every_n_epoch: int = 1  # has to be 1 because of schedulers
    val_split: Optional[ValSplit] = ValSplit.devries
    do_val: bool = False
    batch_size: int = 128
    resume_from_ckpt: bool = False
    benchmark: bool = True  # set to false if input size varies during training!
    fast_dev_run: bool = False  # True/Fals
    lr_scheduler: LRSchedulerConfig = LRSchedulerConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    callbacks: dict[Any, Any] = field(
        default_factory=lambda: {}
    )  # TODO: validate existence

    @validator("num_steps")
    def validate_steps(cls, num_steps: Optional[int], values: dict[str, Any]):
        if (num_steps is None and values["num_epochs"] is None) or (
            num_steps == 0 and values["num_epochs"] == 0
        ):
            raise ValueError("Must specify either num_steps or num_epochs")
        return num_steps


@dataclass
class NetworkConfig:
    name: str = "vgg13"
    backbone: Optional[str] = None
    imagenet_weights_path: Optional[Path] = None
    load_dg_backbone_path: Optional[Path] = None
    save_dg_backbone_path: Optional[Path] = Path("${exp.dir}/dg_backbone.ckpt")

    @validator("name", "backbone")
    def validate_network_name(cls, name: str):
        if name is not None and not networks.network_exists(name):
            raise ValueError(f'Network "{name}" does not exist.')
        return name


@dataclass
class ModelConfig:
    name: str = "devries_model"
    fc_dim: int = 512
    dg_reward: float = 2.2
    avg_pool: bool = True
    dropout_rate: int = 0  # TODO: this should really be a boolean
    monitor_mcd_samples: int = (
        50  # only activated if "mcd" substring in train or val monitor confids.
    )
    test_mcd_samples: int = 50  # only activated if "mcd" substring in test confids.
    budget: float = 0.3
    network: NetworkConfig = NetworkConfig()

    @validator("name")
    def validate_network_name(cls, name: str):
        if name is not None and not models.model_exists(name):
            raise ValueError(f'Model "{name}" does not exist.')
        return name


@dataclass
class PerfMetricsConfig:
    # TODO: Validate Perf metrics
    train: list[str] = field(
        default_factory=lambda: [
            "loss",
            "nll",
            "accuracy",
        ]
    )  # train brier_score logging costs around 5% performance
    val: list[str] = field(
        default_factory=lambda: ["loss", "nll", "accuracy", "brier_score"]
    )
    test: list[str] = field(default_factory=lambda: ["nll", "accuracy", "brier_score"])


@dataclass
class ConfidMetricsConfig:
    train: list[str] = field(
        default_factory=lambda: [
            "failauc",
            "failap_suc",
            "failap_err",
            "fpr@95tpr",
            "e-aurc",
            "aurc",
        ]
    )
    val: list[str] = field(
        default_factory=lambda: [
            "failauc",
            "failap_suc",
            "failap_err",
            "fpr@95tpr",
            "e-aurc",
            "aurc",
        ]
    )
    test: list[str] = field(
        default_factory=lambda: [
            "failauc",
            "failap_suc",
            "failap_err",
            "mce",
            "ece",
            "e-aurc",
            "aurc",
            "fpr@95tpr",
        ]
    )

    @validator("train", "val", "test", each_item=True)
    def validate(cls, name: str):
        if not metrics.metric_function_exists(name):
            raise ValueError(f'Confid metric function "{name}" does not exist.')
        return name


@dataclass
class ConfidMeasuresConfig:
    train: list[str] = field(
        default_factory=lambda: ["det_mcp"]
    )  # mcd_confs not available due to performance. 'det_mcp' costs around 3% (hard to say more volatile)
    val: list[str] = field(
        default_factory=lambda: ["det_mcp"]
    )  # , "mcd_mcp", "mcd_pe", "mcd_ee", "mcd_mi", "mcd_sv"
    test: list[str] = field(default_factory=lambda: ["det_mcp", "det_pe", "ext"])

    @validator("train", "val", "test", each_item=True)
    def validate(cls, name: str):
        if not confid_scores.confid_function_exists(name):
            raise ValueError(f'Confid function "{name}" does not exist.')
        return name


@dataclass
class QueryStudiesConfig:
    iid_study: str = "cifar10"
    noise_study: list[str] = field(default_factory=lambda: ["corrupt_cifar10"])
    new_class_study: list[str] = field(
        default_factory=lambda: [
            "tinyimagenet_resize",
            "cifar100",
            "svhn",
        ]
    )

    @validator("iid_study", "noise_study", "new_class_study", each_item=True)
    def validate(cls, name: str):
        if not dataset_collection.dataset_exists(name):
            raise ValueError(f'Dataset "{name}" does not exist.')
        return name


@dataclass
class EvalConfig:
    performance_metrics: PerfMetricsConfig = PerfMetricsConfig()
    confid_metrics: ConfidMetricsConfig = ConfidMetricsConfig()
    confidence_measures: ConfidMeasuresConfig = ConfidMeasuresConfig()

    monitor_plots: list[str] = field(
        default_factory=lambda: [
            # "overconfidence",
            "hist_per_confid",
        ]
    )

    tb_hparams: list[str] = field(default_factory=lambda: ["fold"])
    ext_confid_name: str = "dg"
    test_conf_scaling: bool = False
    val_tuning: bool = True
    r_star: float = 0.25
    r_delta: float = 0.05

    query_studies: QueryStudiesConfig = QueryStudiesConfig()


@dataclass
class TestConfig:
    name: str = "test_results"
    dir: Path = Path("${exp.dir}/${test.name}")
    cf_path: Path = Path("${exp.dir}/hydra/config.yaml")
    selection_criterion: str = "latest"
    best_ckpt_path: Path = Path("${exp.version_dir}/${test.selection_criterion}.ckpt")
    only_latest_version: bool = True  # if false looks for best metrics across all versions in exp_dir. Turn to false if resumed training.
    devries_repro_ood_split: bool = False
    assim_ood_norm_flag: bool = False
    iid_set_split: str = "devries"  # all, devries
    raw_output_path: str = "raw_output.npz"
    external_confids_output_path: str = "external_confids.npz"
    selection_mode: str = "max" # model selection criterion or "latest"


@dataclass
class DataConfig:
    dataset: str = "cifar10"
    data_dir: Path = Path("${oc.env:DATASET_ROOT_DIR}/${data.dataset}")
    pin_memory: bool = True
    img_size: tuple[int, int, int] = (32, 32, 3)
    num_workers: int = 12
    num_classes: int = 10
    reproduce_confidnet_splits: bool = True
    augmentations: Any = MISSING
    # train: # careful, the order here will determine the order of transforms (except normalize will be executed manually at the end after toTensor)
    #   random_crop: [32, 4] # size, padding
    #   hflip: True
    #   #      rotate: 15
    #   to_tensor:
    #   normalize: [[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]]
    #   cutout: 16
    # val:
    #   to_tensor:
    #   normalize: [[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]]
    # test:
    #   to_tensor:
    #   normalize: [[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]]
    kwargs: Optional[dict[Any, Any]] = None


@dataclass
class Config:
    data: DataConfig = DataConfig()

    trainer: TrainerConfig = TrainerConfig()

    exp: ExperimentConfig = ExperimentConfig()
    model: ModelConfig = ModelConfig()

    eval: EvalConfig = EvalConfig()
    test: TestConfig = TestConfig()


def init():
    store = ConfigStore.instance()
    store.store(name="config_schema", node=Config)
    store.store(group="data", name="data_schema", node=DataConfig)

    store.store(
        group="trainer/lr_scheduler",
        name="LinearWarmupCosineAnnealingLR",
        node=LinearWarmupCosineAnnealingLR,
    )

    store.store(
        group="trainer/lr_scheduler",
        name="CosineAnnealingLR",
        node=CosineAnnealingLR,
    )

    store.store(
        group="trainer/optimizer",
        name="SGD",
        node=SGD,
    )
