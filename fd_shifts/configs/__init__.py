from __future__ import annotations

import importlib
import os
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from random import randint
from typing import TYPE_CHECKING, Any, Iterable, Optional, TypeVar

from omegaconf import SI, DictConfig, OmegaConf

from fd_shifts import get_version

from .iterable_mixin import _IterableMixin

if TYPE_CHECKING:
    import torch
    from pydantic.dataclasses import Dataclass

    ConfigT = TypeVar("ConfigT", bound=Dataclass)


class StrEnum(str, Enum):
    """Enum where members are also (and must be) strings."""

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


@dataclass
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


@dataclass
class OutputPathsPerMode(_IterableMixin):
    """Container for per-mode output paths"""

    fit: OutputPathsConfig = field(
        default_factory=lambda: OutputPathsConfig(
            raw_output=Path("${exp.version_dir}/raw_output.npz"),
            raw_output_dist=Path("${exp.version_dir}/raw_output_dist.npz"),
            external_confids=Path("${exp.version_dir}/external_confids.npz"),
            external_confids_dist=Path("${exp.version_dir}/external_confids_dist.npz"),
            input_imgs_plot=Path("${exp.dir}/input_imgs.png"),
            encoded_output=Path("${test.dir}/encoded_output.npz"),
            encoded_train=Path("${test.dir}/train_features.npz"),
            attributions_output=Path("${test.dir}/attributions.csv"),
        )
    )
    test: OutputPathsConfig = field(
        default_factory=lambda: OutputPathsConfig(
            raw_output=Path("${test.dir}/raw_logits.npz"),
            raw_output_dist=Path("${test.dir}/raw_logits_dist.npz"),
            external_confids=Path("${test.dir}/external_confids.npz"),
            external_confids_dist=Path("${test.dir}/external_confids_dist.npz"),
            input_imgs_plot=None,
            encoded_output=Path("${test.dir}/encoded_output.npz"),
            encoded_train=Path("${test.dir}/train_features.npz"),
            attributions_output=Path("${test.dir}/attributions.csv"),
        )
    )
    analysis: Path = SI("${test.dir}")


@dataclass
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
    output_paths: OutputPathsPerMode = field(
        default_factory=lambda: OutputPathsPerMode()
    )


# @defer_validation
# @dataclass(config=ConfigDict(validate_assignment=True))
# class LRSchedulerConfig:
#     """Base class for LR scheduler configuration"""

#     _target_: str = MISSING
#     _partial_: Optional[bool] = None


# CosineAnnealingLR = builds(
#     torch.optim.lr_scheduler.CosineAnnealingLR,
#     builds_bases=(LRSchedulerConfig,),
#     zen_partial=True,
#     populate_full_signature=True,
#     T_max="${trainer.num_steps}",
# )

# LinearWarmupCosineAnnealingLR = builds(
#     pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR,
#     builds_bases=(LRSchedulerConfig,),
#     zen_partial=True,
#     populate_full_signature=True,
#     max_epochs="${trainer.num_steps}",
#     warmup_epochs=500,
# )


# @defer_validation
# @dataclass(config=ConfigDict(validate_assignment=True))
# class OptimizerConfig:
#     """Base class for optimizer configuration"""

#     _target_: str = MISSING
#     _partial_: Optional[bool] = True


# @defer_validation
# @dataclass(config=ConfigDict(validate_assignment=True))
# class SGD(OptimizerConfig):
#     """Configuration for SGD optimizer"""

#     _target_: str = "torch.optim.sgd.SGD"
#     lr: float = 0.003  # pylint: disable=invalid-name
#     dampening: float = 0.0
#     momentum: float = 0.9
#     nesterov: bool = False
#     maximize: bool = False
#     weight_decay: float = 0.0


# @defer_validation
# @dataclass(config=ConfigDict(validate_assignment=True))
# class Adam(OptimizerConfig):
#     """Configuration for ADAM optimizer"""

#     _target_: str = "torch.optim.adam.Adam"
#     lr: float = 0.003  # pylint: disable=invalid-name
#     betas: tuple[float, float] = (0.9, 0.999)
#     eps: float = 1e-08
#     maximize: bool = False
#     weight_decay: float = 0.0


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


@dataclass
class TrainerConfig(_IterableMixin):
    """Main configuration for PyTorch Lightning Trainer"""

    num_epochs: Optional[int] = 300
    num_steps: Optional[int] = None
    num_epochs_backbone: Optional[int] = None
    val_every_n_epoch: int = 5
    do_val: bool = True
    batch_size: int = 128
    resume_from_ckpt: bool = False
    benchmark: bool = False
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


@dataclass
class NetworkConfig(_IterableMixin):
    """Model Network configuration"""

    name: str = "vgg13"
    backbone: Optional[str] = None
    imagenet_weights_path: Optional[Path] = None
    load_dg_backbone_path: Optional[Path] = None
    save_dg_backbone_path: Optional[Path] = None


@dataclass
class ModelConfig(_IterableMixin):
    """Model Configuration"""

    name: str = "devries_model"
    network: NetworkConfig = field(default_factory=lambda: NetworkConfig())
    fc_dim: int = 512
    avg_pool: bool = True
    dropout_rate: int = 0
    monitor_mcd_samples: int = 50
    test_mcd_samples: int = 50
    confidnet_fc_dim: int | None = None
    dg_reward: float | None = None
    balanced_sampeling: bool = False
    budget: float = 0.3


@dataclass
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


@dataclass
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


@dataclass
class ConfidMeasuresConfig(_IterableMixin):
    """Confidence Measures Configuration"""

    train: list[str] = field(default_factory=lambda: ["det_mcp"])
    val: list[str] = field(default_factory=lambda: ["det_mcp"])
    test: list[str] = field(default_factory=lambda: ["det_mcp", "det_pe"])


@dataclass
class QueryStudiesConfig(_IterableMixin):
    """Query Studies Configuration"""

    iid_study: str | None = None
    noise_study: DataConfig = field(default_factory=lambda: DataConfig())
    in_class_study: list[DataConfig] = field(default_factory=lambda: [])
    new_class_study: list[DataConfig] = field(default_factory=lambda: [])


@dataclass
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


@dataclass
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


@dataclass
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
    target_transforms: Any | None = None
    subsample_corruptions: int = 10
    kwargs: dict[Any, Any] | None = None


@dataclass
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
