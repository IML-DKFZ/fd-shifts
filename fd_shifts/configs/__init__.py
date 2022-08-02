from dataclasses import field
from enum import Enum, auto
from pathlib import Path
from random import randint, random
from typing import Any, Iterator, Optional, TypeVar, cast

import hydra
import pl_bolts
import torch
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra_zen import ZenField, builds
from omegaconf import DictConfig
from omegaconf.omegaconf import MISSING, OmegaConf
from pydantic import validator
from pydantic.dataclasses import dataclass

import fd_shifts
from fd_shifts import models
from fd_shifts.analysis import confid_scores, metrics
from fd_shifts.loaders import dataset_collection

from ..models import networks

# TODO: Clean up data configs (-> instantiation? enum?)
# TODO: Clean up model configs (-> instantiation? enum?)

class AutoName(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name


class Mode(AutoName):
    train = auto()
    test = auto()
    train_test = auto()
    analysis = auto()
    debug = auto()


class ValSplit(AutoName):
    devries = auto()
    repro_confidnet = auto()
    cv = auto()
    zhang = auto()  # TODO: Should this still be here?


class IterableMixin:
    def __iter__(self) -> Iterator[tuple[str, Any]]:
        return filter(
            lambda item: item[0] != "__initialised__", self.__dict__.items()
        ).__iter__()


T = TypeVar("T")


def defer_validation(original_class: type[T]) -> type[T]:
    def __validate(obj):
        obj.__defered_validate()

        for subobj in obj.__dict__.values():
            if hasattr(subobj, "__defered_validate"):
                subobj.validate()

    original_class.__defered_validate = original_class.__post_init__
    original_class.__post_init__ = lambda _: None
    original_class.validate = __validate
    return original_class


@defer_validation
@dataclass
class OutputPathsConfig(IterableMixin):
    input_imgs_plot: Optional[Path] = None
    raw_output: Path = MISSING
    raw_output_dist: Path = MISSING
    external_confids: Path = MISSING
    external_confids_dist: Path = MISSING


@defer_validation
@dataclass
class OutputPathsPerMode(IterableMixin):
    fit: OutputPathsConfig = OutputPathsConfig()
    test: OutputPathsConfig = OutputPathsConfig(
        input_imgs_plot=MISSING,
        raw_output=MISSING,
        raw_output_dist=MISSING,
        external_confids=MISSING,
        external_confids_dist=MISSING,
    )


@defer_validation
@dataclass
class ExperimentConfig(IterableMixin):
    group_name: str = MISSING
    name: str = MISSING
    version: Optional[int] = None
    mode: Mode = MISSING
    work_dir: Path = MISSING
    fold_dir: Path = MISSING
    root_dir: Path = MISSING
    data_root_dir: Path = MISSING
    group_dir: Path = MISSING
    dir: Path = MISSING
    version_dir: Path = MISSING
    fold: int = MISSING
    crossval_n_folds: int = MISSING
    crossval_ids_path: Path = MISSING
    output_paths: OutputPathsPerMode = OutputPathsPerMode()
    log_path: Path = MISSING
    global_seed: int = MISSING


@defer_validation
@dataclass
class LRSchedulerConfig:
    _target_: str = MISSING
    _partial_: Optional[str] = None

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


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


@defer_validation
@dataclass
class OptimizerConfig:
    _target_: str = MISSING
    _partial_: Optional[str] = None

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


SGD = builds(
    torch.optim.SGD,
    lr=0.003,
    momentum=0.9,
    weight_decay=0.0,
    builds_bases=(OptimizerConfig,),
    zen_partial=True,
    populate_full_signature=True,
)


@defer_validation
@dataclass
class TrainerConfig(IterableMixin):
    resume_from_ckpt_confidnet: Optional[bool] = None
    num_epochs: Optional[int] = None
    num_steps: Optional[int] = None
    num_epochs_backbone: Optional[int] = None
    dg_pretrain_epochs: Optional[int] = None
    val_every_n_epoch: int = MISSING
    val_split: Optional[ValSplit] = None
    do_val: bool = MISSING
    batch_size: int = MISSING
    resume_from_ckpt: bool = MISSING
    benchmark: bool = MISSING
    fast_dev_run: bool = MISSING
    lr_scheduler: LRSchedulerConfig = LRSchedulerConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    callbacks: dict[str, Optional[dict[Any, Any]]] = field(
        default_factory=lambda: {}
    )  # TODO: validate existence

    learning_rate_confidnet: Optional[float] = None
    learning_rate_confidnet_finetune: Optional[float] = None

    @validator("num_steps")
    def validate_steps(cls, num_steps: Optional[int], values: dict[str, Any]):
        if (num_steps is None and values["num_epochs"] is None) or (
            num_steps == 0 and values["num_epochs"] == 0
        ):
            raise ValueError("Must specify either num_steps or num_epochs")
        return num_steps


@defer_validation
@dataclass
class NetworkConfig(IterableMixin):
    name: str = MISSING
    backbone: Optional[str] = None
    imagenet_weights_path: Optional[Path] = None
    load_dg_backbone_path: Optional[Path] = None
    save_dg_backbone_path: Optional[Path] = None

    @validator("name", "backbone")
    def validate_network_name(cls, name: str):
        if name is not None and not networks.network_exists(name):
            raise ValueError(f'Network "{name}" does not exist.')
        return name


@defer_validation
@dataclass
class ModelConfig(IterableMixin):
    name: str = MISSING
    fc_dim: int = MISSING
    confidnet_fc_dim: Optional[int] = None
    dg_reward: Optional[float] = None
    avg_pool: bool = MISSING
    dropout_rate: int = MISSING
    monitor_mcd_samples: int = MISSING
    test_mcd_samples: int = MISSING
    budget: Optional[float] = None
    network: NetworkConfig = NetworkConfig()

    @validator("name")
    def validate_network_name(cls, name: str):
        if name is not None and not models.model_exists(name):
            raise ValueError(f'Model "{name}" does not exist.')
        return name


@defer_validation
@dataclass
class PerfMetricsConfig(IterableMixin):
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


@defer_validation
@dataclass
class ConfidMetricsConfig(IterableMixin):
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


@defer_validation
@dataclass
class ConfidMeasuresConfig(IterableMixin):
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


@defer_validation
@dataclass
class QueryStudiesConfig(IterableMixin):
    iid_study: str = MISSING
    noise_study: list[str] = MISSING
    in_class_study: list[str] = MISSING
    new_class_study: list[str] = MISSING

    @validator(
        "iid_study", "in_class_study", "noise_study", "new_class_study", each_item=True
    )
    def validate(cls, name: str):
        if not dataset_collection.dataset_exists(name):
            raise ValueError(f'Dataset "{name}" does not exist.')
        return name


@defer_validation
@dataclass
class EvalConfig(IterableMixin):
    performance_metrics: PerfMetricsConfig = PerfMetricsConfig()
    confid_metrics: ConfidMetricsConfig = ConfidMetricsConfig()
    confidence_measures: ConfidMeasuresConfig = ConfidMeasuresConfig()

    monitor_plots: list[str] = field(
        default_factory=lambda: [
            # "overconfidence",
            "hist_per_confid",
        ]
    )

    tb_hparams: list[str] = MISSING
    ext_confid_name: Optional[str] = None
    test_conf_scaling: bool = MISSING
    val_tuning: bool = MISSING
    r_star: float = MISSING
    r_delta: float = MISSING

    query_studies: QueryStudiesConfig = QueryStudiesConfig()


@defer_validation
@dataclass
class TestConfig(IterableMixin):
    name: str = MISSING
    dir: Path = MISSING
    cf_path: Path = MISSING
    selection_criterion: str = MISSING
    best_ckpt_path: Path = MISSING
    only_latest_version: bool = MISSING
    devries_repro_ood_split: bool = MISSING
    assim_ood_norm_flag: bool = MISSING
    iid_set_split: str = MISSING
    raw_output_path: str = MISSING
    external_confids_output_path: str = MISSING
    selection_mode: Optional[str] = None
    output_precision: int = MISSING


@defer_validation
@dataclass
class DataConfig(IterableMixin):
    dataset: str = MISSING
    data_dir: Path = MISSING
    pin_memory: bool = MISSING
    img_size: tuple[int, int, int] = MISSING
    num_workers: int = MISSING
    num_classes: int = MISSING
    reproduce_confidnet_splits: bool = MISSING
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


@defer_validation
@dataclass
class Config(IterableMixin):
    pkgversion: str = MISSING
    data: DataConfig = DataConfig()

    trainer: TrainerConfig = TrainerConfig()

    exp: ExperimentConfig = ExperimentConfig()
    model: ModelConfig = ModelConfig()

    eval: EvalConfig = EvalConfig()
    test: TestConfig = TestConfig()

    def validate(self):
        pass

    @validator("pkgversion")
    def validate_version(cls, version: str):
        if version != fd_shifts.version():
            raise ValueError(
                f"This config was created with version {version} of fd-shifts. "
                f"You are on {fd_shifts.version()}."
            )

        return version


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


def dictconfig_to_object(dcfg: DictConfig) -> Config:
    cfg: Config = cast(Config, OmegaConf.to_object(dcfg))  # only affects the linter
    return cfg
