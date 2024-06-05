from __future__ import annotations

import importlib
import os
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import field
from enum import Enum, auto
from pathlib import Path
from random import randint
from typing import TYPE_CHECKING, Any, Iterable, Optional, TypeVar

from hydra.core.config_store import ConfigStore
from omegaconf import SI, DictConfig, OmegaConf
from pydantic import ConfigDict, validator
from pydantic.dataclasses import dataclass
from typing_extensions import dataclass_transform

from fd_shifts import get_version

from .iterable_mixin import _IterableMixin

if TYPE_CHECKING:
    import torch
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

    raw_output: Path
    raw_output_dist: Path
    external_confids: Path
    external_confids_dist: Path
    encoded_output: Path
    encoded_train: Path
    attributions_output: Path
    input_imgs_plot: Optional[Path] = None


@defer_validation
@dataclass(config=ConfigDict(validate_assignment=True))
class OutputPathsPerMode(_IterableMixin):
    """Container for per-mode output paths"""

    fit: OutputPathsConfig = OutputPathsConfig(
        raw_output=Path("${exp.version_dir}/raw_output.npz"),
        raw_output_dist=Path("${exp.version_dir}/raw_output_dist.npz"),
        external_confids=Path("${exp.version_dir}/external_confids.npz"),
        external_confids_dist=Path("${exp.version_dir}/external_confids_dist.npz"),
        input_imgs_plot=Path("${exp.dir}/input_imgs.png"),
        encoded_output=Path("${test.dir}/encoded_output.npz"),
        encoded_train=Path("${test.dir}/train_features.npz"),
        attributions_output=Path("${test.dir}/attributions.csv"),
    )
    test: OutputPathsConfig = OutputPathsConfig(
        raw_output=Path("${test.dir}/raw_logits.npz"),
        raw_output_dist=Path("${test.dir}/raw_logits_dist.npz"),
        external_confids=Path("${test.dir}/external_confids.npz"),
        external_confids_dist=Path("${test.dir}/external_confids_dist.npz"),
        input_imgs_plot=None,
        encoded_output=Path("${test.dir}/encoded_output.npz"),
        encoded_train=Path("${test.dir}/train_features.npz"),
        attributions_output=Path("${test.dir}/attributions.csv"),
    )
    analysis: Path = SI("${test.dir}")


@defer_validation
@dataclass(config=ConfigDict(validate_assignment=True))
class ExperimentConfig(_IterableMixin):
    """Main experiment config"""

    group_name: str
    name: str
    mode: Mode = Mode.train_test
    work_dir: Path = Path.cwd()
    fold_dir: Path = Path("exp/${exp.fold}")
    root_dir: Path | None = Path(p) if (p := os.getenv("EXPERIMENT_ROOT_DIR")) else None
    data_root_dir: Path | None = (
        Path(p) if (p := os.getenv("DATASET_ROOT_DIR")) else None
    )
    group_dir: Path = Path("${exp.root_dir}/${exp.group_name}")
    dir: Path = Path("${exp.group_dir}/${exp.name}")
    version: int | None = None
    version_dir: Path = Path("${exp.dir}/version_${exp.version}")
    fold: int = 0
    crossval_n_folds: int = 10
    crossval_ids_path: Path = Path("${exp.dir}/crossval_ids.pickle")
    log_path: Path = Path("log.txt")
    global_seed: int = randint(0, 1_000_000)
    output_paths: OutputPathsPerMode = OutputPathsPerMode()


@dataclass
class LRSchedulerConfig:
    init_args: dict
    class_path: str = "fd_shifts.configs.LRSchedulerConfig"

    def __call__(
        self, optim: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler._LRScheduler:
        module_name, class_name = self.init_args["class_path"].rsplit(".", 1)
        cls = getattr(importlib.import_module(module_name), class_name)
        return cls(optim, **self.init_args["init_args"])


@dataclass
class OptimizerConfig:
    init_args: dict
    class_path: str = "fd_shifts.configs.OptimizerConfig"

    def __call__(self, params: Iterable) -> torch.optim.Optimizer:
        module_name, class_name = self.init_args["class_path"].rsplit(".", 1)
        cls = getattr(importlib.import_module(module_name), class_name)
        return cls(params, **self.init_args["init_args"])


@defer_validation
@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class TrainerConfig(_IterableMixin):
    """Main configuration for PyTorch Lightning Trainer"""

    num_epochs: Optional[int] = 300
    num_steps: Optional[int] = None
    num_epochs_backbone: Optional[int] = None
    val_every_n_epoch: int = 5
    do_val: bool = True
    batch_size: int = 128
    resume_from_ckpt: bool = False
    benchmark: bool = True
    fast_dev_run: bool | int = False
    # lr_scheduler: Callable[
    #     [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
    # ] | None = None
    # optimizer: Callable[[Iterable], torch.optim.Optimizer] | None = None
    lr_scheduler: LRSchedulerConfig | None = None
    optimizer: OptimizerConfig | None = None
    accumulate_grad_batches: int = 1
    resume_from_ckpt_confidnet: bool = False
    dg_pretrain_epochs: int | None = 100
    dg_pretrain_steps: Optional[int] = None
    val_split: ValSplit = ValSplit.devries
    lr_scheduler_interval: str = "epoch"

    # TODO: Replace with jsonargparse compatible type hint to lightning.Callback
    callbacks: dict[str, Optional[dict[Any, Any]]] = field(
        default_factory=lambda: {
            "model_checkpoint": None,
            "confid_monitor": None,
            "learning_rate_monitor": None,
        }
    )

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

    name: str = "vgg13"
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
        from ..models import networks

        if name is not None and not networks.network_exists(name):
            raise ValueError(f'Network "{name}" does not exist.')
        return name


@defer_validation
@dataclass(config=ConfigDict(validate_assignment=True))
class ModelConfig(_IterableMixin):
    """Model Configuration"""

    name: str = "devries_model"
    network: NetworkConfig = field(default_factory=lambda: NetworkConfig())
    fc_dim: int = 512
    avg_pool: bool = True
    dropout_rate: int = 0
    monitor_mcd_samples: int = 50
    test_mcd_samples: int = 50
    confidnet_fc_dim: Optional[int] = None
    dg_reward: Optional[float] = None
    balanced_sampeling: bool = False
    budget: float = 0.3
    clip_class_prefix: Optional[str] = None

    # pylint: disable=no-self-argument
    @validator("name")
    def validate_network_name(cls: ModelConfig, name: str) -> str:
        """Check if the model exists

            cls (ModelConfig):
            name (str):

        Returns:
            name
        """
        from fd_shifts import models

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
        from fd_shifts.analysis import metrics

        if not metrics.metric_function_exists(name):
            raise ValueError(f'Confid metric function "{name}" does not exist.')
        return name


@defer_validation
@dataclass(config=ConfigDict(validate_assignment=True))
class ConfidMeasuresConfig(_IterableMixin):
    """Confidence Measures Configuration"""

    train: list[str] = field(default_factory=lambda: ["det_mcp"])
    val: list[str] = field(default_factory=lambda: ["det_mcp"])
    test: list[str] = field(default_factory=lambda: ["det_mcp", "det_pe"])

    # pylint: disable=no-self-argument
    @validator("train", "val", "test", each_item=True)
    def validate(cls: ConfidMeasuresConfig, name: str) -> str:
        """Check all confid functions exist
            cls (type[ConfidMeasuresConfig]):
            name (str):

        Returns:
            name
        """
        from fd_shifts.analysis import confid_scores

        if not confid_scores.confid_function_exists(name):
            raise ValueError(f'Confid function "{name}" does not exist.')
        return name


@defer_validation
@dataclass(config=ConfigDict(validate_assignment=True))
class QueryStudiesConfig(_IterableMixin):
    """Query Studies Configuration"""

    iid_study: str | None = None
    noise_study: DataConfig = field(default_factory=lambda: DataConfig())
    in_class_study: list[DataConfig] = field(default_factory=lambda: [])
    new_class_study: list[DataConfig] = field(default_factory=lambda: [])

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
        from fd_shifts.loaders import dataset_collection

        if not dataset_collection.dataset_exists(name):
            raise ValueError(f'Dataset "{name}" does not exist.')
        return name


@defer_validation
@dataclass(config=ConfigDict(validate_assignment=True))
class EvalConfig(_IterableMixin):
    """Evaluation Configuration container"""

    tb_hparams: list[str] = field(default_factory=lambda: ["fold"])
    test_conf_scaling: bool = False
    val_tuning: bool = True
    r_star: float = 0.25
    r_delta: float = 0.05

    query_studies: QueryStudiesConfig = field(
        default_factory=lambda: QueryStudiesConfig()
    )
    performance_metrics: PerfMetricsConfig = field(
        default_factory=lambda: PerfMetricsConfig()
    )
    confid_metrics: ConfidMetricsConfig = field(
        default_factory=lambda: ConfidMetricsConfig()
    )
    confidence_measures: ConfidMeasuresConfig = field(
        default_factory=lambda: ConfidMeasuresConfig()
    )

    monitor_plots: list[str] = field(
        default_factory=lambda: [
            "hist_per_confid",
        ]
    )

    ext_confid_name: Optional[str] = None


@defer_validation
@dataclass(config=ConfigDict(validate_assignment=True))
class TestConfig(_IterableMixin):
    """Inference time configuration"""

    name: str = "test_results"
    dir: Path = Path("${exp.dir}/${test.name}")
    cf_path: Path = Path("${exp.dir}/hydra/config.yaml")
    selection_criterion: str = "latest"
    best_ckpt_path: Path = Path("${exp.version_dir}/${test.selection_criterion}.ckpt")
    only_latest_version: bool = True
    devries_repro_ood_split: bool = False
    assim_ood_norm_flag: bool = False
    iid_set_split: str = "devries"
    raw_output_path: str = "raw_output.npz"
    external_confids_output_path: str = "external_confids.npz"
    output_precision: int = 16
    selection_mode: Optional[str] = "max"
    compute_train_encodings: bool = False


@defer_validation
@dataclass(config=ConfigDict(validate_assignment=True))
class DataConfig(_IterableMixin):
    """Dataset Configuration"""

    dataset: str | None = None
    data_dir: Path | None = None
    pin_memory: bool = True
    img_size: tuple[int, int, int] | None = None
    num_workers: int = 12
    num_classes: int | None = None
    reproduce_confidnet_splits: bool = False
    augmentations: dict[str, dict[str, Any]] | None = None
    target_transforms: Optional[Any] = None
    subsample_corruptions: int = 10
    kwargs: Optional[dict[Any, Any]] = None


@defer_validation
@dataclass(config=ConfigDict(validate_assignment=True))
class Config(_IterableMixin):
    """Main Configuration Class"""

    exp: ExperimentConfig

    pkgversion: str = get_version()

    data: DataConfig = field(default_factory=lambda: DataConfig())

    trainer: TrainerConfig = field(default_factory=lambda: TrainerConfig())

    model: ModelConfig = field(default_factory=lambda: ModelConfig())

    eval: EvalConfig = field(default_factory=lambda: EvalConfig())
    test: TestConfig = field(default_factory=lambda: TestConfig())

    def update_experiment(self, name: str):
        from fd_shifts.utils import exp_utils

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
