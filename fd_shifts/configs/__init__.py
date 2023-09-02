from __future__ import annotations

import os
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import field
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Optional, TypeVar

import pl_bolts
import torch
from hydra.core.config_store import ConfigStore
from hydra_zen import builds  # type: ignore
from omegaconf import DictConfig, OmegaConf
from omegaconf.omegaconf import MISSING
from pydantic import ConfigDict, validator
from pydantic.dataclasses import dataclass
from typing_extensions import dataclass_transform

from fd_shifts import models
from fd_shifts.analysis import confid_scores, metrics
from fd_shifts.loaders import dataset_collection
from fd_shifts.utils import exp_utils

from ..models import networks

if TYPE_CHECKING:
    from pydantic.dataclasses import Dataclass

    ConfigT = TypeVar("ConfigT", bound=Dataclass)


class StrEnum(str, Enum):
    """Enum where members are also (and must be) strings"""

    # pylint: disable=no-self-argument
    def _generate_next_value_(name, start, count, last_values):  # type: ignore
        return name.lower()


class Mode(StrEnum):
    """Experiment mode"""

    train = auto()
    test = auto()
    train_test = auto()
    analysis = auto()
    debug = auto()


class ValSplit(StrEnum):
    """Ways to split off a validation set"""

    devries = auto()
    repro_confidnet = auto()
    cv = auto()
    zhang = auto()


class _IterableMixin:  # pylint: disable=too-few-public-methods
    def __iter__(self) -> Iterator[tuple[str, Any]]:
        return filter(
            lambda item: not item[0].startswith("__"), self.__dict__.items()
        ).__iter__()


@dataclass_transform()
def defer_validation(original_class: type[ConfigT]) -> type[ConfigT]:
    """Disable validation for a pydantic dataclass

        original_class (type[T]): original pydantic dataclass

    Returns:
        original_class but with validation disabled
    """
    original_class.__pydantic_run_validation__ = False
    return original_class


@defer_validation
@dataclass(config=ConfigDict(validate_assignment=True))
class OutputPathsConfig(_IterableMixin):
    """Where outputs are stored"""

    input_imgs_plot: Optional[Path] = None
    raw_output: Path = MISSING
    encoded_output: Optional[Path] = None
    attributions_output: Optional[Path] = None
    raw_output_dist: Path = MISSING
    external_confids: Path = MISSING
    external_confids_dist: Path = MISSING


@defer_validation
@dataclass(config=ConfigDict(validate_assignment=True))
class OutputPathsPerMode(_IterableMixin):
    """Container for per-mode output paths"""

    fit: OutputPathsConfig = OutputPathsConfig()
    test: OutputPathsConfig = OutputPathsConfig()


@defer_validation
@dataclass(config=ConfigDict(validate_assignment=True))
class ExperimentConfig(_IterableMixin):
    """Main experiment config"""

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
@dataclass(config=ConfigDict(validate_assignment=True))
class LRSchedulerConfig:
    """Base class for LR scheduler configuration"""

    _target_: str = MISSING
    _partial_: Optional[bool] = None


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
@dataclass(config=ConfigDict(validate_assignment=True))
class OptimizerConfig:
    """Base class for optimizer configuration"""

    _target_: str = MISSING
    _partial_: Optional[bool] = True


@defer_validation
@dataclass(config=ConfigDict(validate_assignment=True))
class SGD(OptimizerConfig):
    """Configuration for SGD optimizer"""

    _target_: str = "torch.optim.sgd.SGD"
    lr: float = 0.003  # pylint: disable=invalid-name
    dampening: float = 0.0
    momentum: float = 0.9
    nesterov: bool = False
    maximize: bool = False
    weight_decay: float = 0.0


@defer_validation
@dataclass(config=ConfigDict(validate_assignment=True))
class Adam(OptimizerConfig):
    """Configuration for ADAM optimizer"""

    _target_: str = "torch.optim.adam.Adam"
    lr: float = 0.003  # pylint: disable=invalid-name
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-08
    maximize: bool = False
    weight_decay: float = 0.0


@defer_validation
@dataclass(config=ConfigDict(validate_assignment=True))
class TrainerConfig(_IterableMixin):
    """Main configuration for PyTorch Lightning Trainer"""

    accumulate_grad_batches: int = 1
    resume_from_ckpt_confidnet: Optional[bool] = None
    num_epochs: Optional[int] = None
    num_steps: Optional[int] = None
    num_epochs_backbone: Optional[int] = None
    dg_pretrain_epochs: Optional[int] = None
    dg_pretrain_steps: Optional[int] = None
    val_every_n_epoch: int = MISSING
    val_split: Optional[ValSplit] = None
    do_val: bool = MISSING
    batch_size: int = MISSING
    resume_from_ckpt: bool = MISSING
    benchmark: bool = MISSING
    fast_dev_run: bool | int = MISSING
    lr_scheduler_interval: str = "epoch"
    lr_scheduler: LRSchedulerConfig = LRSchedulerConfig()
    optimizer: OptimizerConfig = MISSING
    callbacks: dict[str, Optional[dict[Any, Any]]] = field(default_factory=lambda: {})

    learning_rate_confidnet: Optional[float] = None
    learning_rate_confidnet_finetune: Optional[float] = None

    # pylint: disable=no-self-argument
    @validator("num_steps")
    def validate_steps(
        cls: TrainerConfig, num_steps: Optional[int], values: dict[str, Any]
    ) -> Optional[int]:
        """Validate either num_epochs or num_steps is set

            cls (TrainerConfig): TrainerConfig
            num_steps (Optional[int]): num_steps value
            values (dict[str, Any]): other values

        Returns:
            num_steps
        """
        if (num_steps is None and values["num_epochs"] is None) or (
            num_steps == 0 and values["num_epochs"] == 0
        ):
            raise ValueError("Must specify either num_steps or num_epochs")
        return num_steps


@defer_validation
@dataclass(config=ConfigDict(validate_assignment=True))
class NetworkConfig(_IterableMixin):
    """Model Network configuration"""

    name: str = MISSING
    backbone: Optional[str] = None
    imagenet_weights_path: Optional[Path] = None
    load_dg_backbone_path: Optional[Path] = None
    save_dg_backbone_path: Optional[Path] = None

    # pylint: disable=no-self-argument
    @validator("name", "backbone")
    def validate_network_name(cls: NetworkConfig, name: str) -> str:
        """Check if network and backbone exist

            cls (NetworkConfig): this config
            name (str): name of the network

        Returns:
            name
        """
        if name is not None and not networks.network_exists(name):
            raise ValueError(f'Network "{name}" does not exist.')
        return name


@defer_validation
@dataclass(config=ConfigDict(validate_assignment=True))
class ModelConfig(_IterableMixin):
    """Model Configuration"""

    name: str = MISSING
    fc_dim: int = MISSING
    confidnet_fc_dim: Optional[int] = None
    dg_reward: Optional[float] = None
    avg_pool: bool = MISSING
    balanced_sampeling: bool = False
    dropout_rate: int = MISSING
    monitor_mcd_samples: int = MISSING
    test_mcd_samples: int = MISSING
    budget: Optional[float] = None
    network: NetworkConfig = NetworkConfig()

    # pylint: disable=no-self-argument
    @validator("name")
    def validate_network_name(cls: ModelConfig, name: str) -> str:
        """Check if the model exists

            cls (ModelConfig):
            name (str):

        Returns:
            name
        """
        if name is not None and not models.model_exists(name):
            raise ValueError(f'Model "{name}" does not exist.')
        return name


@defer_validation
@dataclass(config=ConfigDict(validate_assignment=True))
class PerfMetricsConfig(_IterableMixin):
    """Performance Metrics Configuration"""

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
@dataclass(config=ConfigDict(validate_assignment=True))
class ConfidMetricsConfig(_IterableMixin):
    """Confidence Metrics Configuration"""

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

    # pylint: disable=no-self-argument
    @validator("train", "val", "test", each_item=True)
    def validate(cls: ConfidMetricsConfig, name: str) -> str:
        """Check all metric functions exist

            cls (ConfidMetricsConfig)
            name (str)

        Returns:
            name
        """
        if not metrics.metric_function_exists(name):
            raise ValueError(f'Confid metric function "{name}" does not exist.')
        return name


@defer_validation
@dataclass(config=ConfigDict(validate_assignment=True))
class ConfidMeasuresConfig(_IterableMixin):
    """Confidence Measures Configuration"""

    train: list[str] = field(default_factory=lambda: ["det_mcp"])
    val: list[str] = field(default_factory=lambda: ["det_mcp"])
    test: list[str] = field(default_factory=lambda: ["det_mcp", "det_pe", "ext"])

    # pylint: disable=no-self-argument
    @validator("train", "val", "test", each_item=True)
    def validate(cls: ConfidMeasuresConfig, name: str) -> str:
        """Check all confid functions exist
            cls (type[ConfidMeasuresConfig]):
            name (str):

        Returns:
            name
        """
        if not confid_scores.confid_function_exists(name):
            raise ValueError(f'Confid function "{name}" does not exist.')
        return name


@defer_validation
@dataclass(config=ConfigDict(validate_assignment=True))
class QueryStudiesConfig(_IterableMixin):
    """Query Studies Configuration"""

    iid_study: str = MISSING
    noise_study: list[str] = MISSING
    in_class_study: list[str] = MISSING
    new_class_study: list[str] = MISSING

    # pylint: disable=no-self-argument
    @validator(
        "iid_study", "in_class_study", "noise_study", "new_class_study", each_item=True
    )
    def validate(cls, name: str) -> str:
        """Check all datasets exist
            cls ():
            name (str):

        Returns:
            name
        """
        if not dataset_collection.dataset_exists(name):
            raise ValueError(f'Dataset "{name}" does not exist.')
        return name


@defer_validation
@dataclass(config=ConfigDict(validate_assignment=True))
class EvalConfig(_IterableMixin):
    """Evaluation Configuration container"""

    performance_metrics: PerfMetricsConfig = PerfMetricsConfig()
    confid_metrics: ConfidMetricsConfig = ConfidMetricsConfig()
    confidence_measures: ConfidMeasuresConfig = ConfidMeasuresConfig()

    monitor_plots: list[str] = field(
        default_factory=lambda: [
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
@dataclass(config=ConfigDict(validate_assignment=True))
class TestConfig(_IterableMixin):
    """Inference time configuration"""

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
@dataclass(config=ConfigDict(validate_assignment=True))
class DataConfig(_IterableMixin):
    """Dataset Configuration"""

    dataset: str = MISSING
    data_dir: Path = MISSING
    pin_memory: bool = MISSING
    img_size: tuple[int, int, int] = MISSING
    num_workers: int = MISSING
    num_classes: int = MISSING
    reproduce_confidnet_splits: bool = MISSING
    augmentations: Any = MISSING
    target_transforms: Optional[Any] = None
    kwargs: Optional[dict[Any, Any]] = None


@defer_validation
@dataclass(config=ConfigDict(validate_assignment=True))
class Config(_IterableMixin):
    """Main Configuration Class"""

    pkgversion: str = MISSING
    data: DataConfig = DataConfig()

    trainer: TrainerConfig = TrainerConfig()

    exp: ExperimentConfig = ExperimentConfig()
    model: ModelConfig = ModelConfig()

    eval: EvalConfig = EvalConfig()
    test: TestConfig = TestConfig()

    def update_experiment(self, name: str):
        config = deepcopy(self)
        group_name = config.data.dataset
        group_dir = config.exp.group_dir.parent / group_name
        exp_dir = group_dir / name
        exp_dir.mkdir(exist_ok=True, parents=True)
        version = exp_utils.get_next_version(exp_dir)

        config.exp = ExperimentConfig(
            group_name=group_name,
            name=name,
            mode=Mode.train,
            fold=0,
            crossval_n_folds=0,
            global_seed=1234,
            version=version,
            work_dir=os.getcwd(),
            data_root_dir=os.getenv("DATASET_ROOT_DIR"),
            group_dir=group_dir,
            dir=exp_dir,
            version_dir=exp_dir / f"version_{version}",
            output_paths=OutputPathsPerMode(
                test=OutputPathsConfig(
                    raw_output=exp_dir / "test_results" / "raw_logits.npz",
                    raw_output_dist=exp_dir / "test_results" / "raw_logits_dist.npz",
                    external_confids=exp_dir / "test_results" / "external_confids.npz",
                    external_confids_dist=exp_dir
                    / "test_results"
                    / "external_confids_dist.npz",
                )
            ),
        )

        config.test = TestConfig(
            name="test_results",
            dir=exp_dir / "test_results",
            cf_path=exp_dir / "hydra/config.yaml",
            selection_criterion="latest",
            best_ckpt_path=exp_dir / f"version_{version}/latest.ckpt",
            only_latest_version=True,
            devries_repro_ood_split=False,
            assim_ood_norm_flag=False,
            iid_set_split="devries",
            raw_output_path="raw_logits.npz",
            external_confids_output_path="external_confids.npz",
            selection_mode="max",
            output_precision=64,
        )

        return config

    @classmethod
    def with_defaults(
        cls, study: str = "deepgamblers", data: str = "cifar10", mode: Mode = Mode.debug
    ) -> Config:
        """Create a config object with populated defaults

        Args:
            cls (type): this class
            study: (str): the study to take defaults from
            data: (str): the dataset to take defaults from
            mode: (Mode): the running mode

        Returns:
            the populated config object
        """
        base_config = OmegaConf.load(
            Path(__file__).parent.parent / "configs" / "config.yaml"
        )
        base_config = OmegaConf.to_container(base_config)

        data_config = OmegaConf.load(
            Path(__file__).parent.parent / "configs" / "data" / f"{data}_data.yaml"
        )
        data_config = OmegaConf.to_container(data_config)
        study_config = OmegaConf.load(
            Path(__file__).parent.parent / "configs" / "study" / f"{study}.yaml"
        )
        study_config = OmegaConf.to_container(study_config)

        base_config = _update(base_config, study_config)
        base_config = _update(base_config, data_config)

        base_config["exp"]["work_dir"] = os.getcwd()
        base_config["exp"]["mode"] = mode
        base_config["trainer"]["lr_scheduler"][
            "_target_"
        ] = "torch.optim.lr_scheduler.CosineAnnealingLR"
        base_config["trainer"]["lr_scheduler"]["_partial_"] = True
        base_config["trainer"]["optimizer"]["_target_"] = "torch.optim.sgd.SGD"
        base_config = OmegaConf.to_container(DictConfig(base_config), resolve=True)
        config = Config(**base_config)
        config.__pydantic_validate_values__()

        return config

    # pylint: disable=no-self-argument
    @validator("pkgversion")
    def validate_version(cls, version: str) -> str:
        """Check if the running version is the same as the version of the configuration
            cls ():
            version (str):

        Returns:
            version
        """
        return version


def _update(d, u):
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = _update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def init() -> None:
    """Initialize the hydra config store with config classes"""
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

    store.store(
        group="trainer/optimizer",
        name="Adam",
        node=Adam,
    )
