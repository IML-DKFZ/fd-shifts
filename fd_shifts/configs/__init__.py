from dataclasses import field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Optional

import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf.omegaconf import MISSING
from pydantic import validator
from pydantic.dataclasses import dataclass

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
class TrainerConfig:
    resume_from_ckpt_confidnet: bool = False
    num_epochs: int = 300  # 250 has to be >1 because of incompatibility of lighting eval with psuedo test
    dg_pretrain_epochs: int = 100  # 100 and 300 total epochs
    val_every_n_epoch: int = 1  # has to be 1 because of schedulers
    val_split: Optional[ValSplit] = ValSplit.devries
    do_val: bool = False
    batch_size: int = 128
    resume_from_ckpt: bool = False
    benchmark: bool = True  # set to false if input size varies during training!
    fast_dev_run: bool = False  # True/Fals
    lr_scheduler: Any = MISSING
    #     name: "CosineAnnealing" # "MultiStep" "CosineAnnealing"
    #     milestones: [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275] # lighting only steps schedulre during validation. so milestones need to be divisible by val_every_n_epoch
    #     max_epochs: ${trainer.num_epochs}
    #     gamma: 0.5
    optimizer: Any = MISSING
    #     name: SGD
    #     learning_rate: 1e-1
    #     momentum: 0.9
    #     nesterov: False
    #     weight_decay: 0.0005
    callbacks: Any = MISSING
    #     model_checkpoint:
    #     confid_monitor:
    #     learning_rate_monitor:


@dataclass
class NetworkConfig:
    name: str = "vgg13"
    imagenet_weights_path: Optional[Path] = None
    load_dg_backbone_path: Optional[Path] = None
    save_dg_backbone_path: Path = Path("${exp.dir}/dg_backbone.ckpt")

    @validator("name")
    def validate_network_name(cls, name: str):
        if not networks.network_exists(name):
            raise ValueError(f'Network "{name}" does not exist.')
        return name


@dataclass
class ModelConfig:
    name: str = "devries_model"  # TODO: make this an enum to check existance
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


# @dataclass
# class EvalConfig:
#   performance_metrics:
#     train: ["loss", "nll", "accuracy"] # train brier_score logging costs around 5% performance
#     val: ["loss", "nll", "accuracy", "brier_score"]
#     test: ["nll", "accuracy", "brier_score"]
#   confid_metrics:
#     train:
#       ["failauc", "failap_suc", "failap_err", "fpr@95tpr", "e-aurc", "aurc"]
#     val: ["failauc", "failap_suc", "failap_err", "fpr@95tpr", "e-aurc", "aurc"]
#     test:
#       [
#         "failauc",
#         "failap_suc",
#         "failap_err",
#         "mce",
#         "ece",
#         "e-aurc",
#         "aurc",
#         "fpr@95tpr",
#       ]
#   confidence_measures: # ["det_mcp" , "det_pe", "tcp" , "mcd_mcp", "mcd_pe", "mcd_ee", "mcd_mi", "mcd_sv"]
#     train: ["det_mcp"] # mcd_confs not available due to performance. 'det_mcp' costs around 3% (hard to say more volatile)
#     val: ["det_mcp"] # , "mcd_mcp", "mcd_pe", "mcd_ee", "mcd_mi", "mcd_sv"
#     test: ["det_mcp", "det_pe", "ext"]
#
#   monitor_plots: #"calibration",
#     [
#       #"overconfidence",
#       "hist_per_confid",
#     ]
#
#   tb_hparams: ["fold"]
#   ext_confid_name: "dg"
#   test_conf_scaling: False
#   val_tuning: True
#   r_star: 0.25
#   r_delta: 0.05
#
#   query_studies: # iid_study, new_class_study, sub_class_study, noise_study
#     iid_study: cifar10
#     noise_study:
#       - corrupt_cifar10
#     new_class_study:
#       # - tinyimagenet_resize
#       - cifar100
#       - svhn
#
# @dataclass
# class TestConfig:
#   name: test_results
#   dir: ${exp.dir}/${test.name}
#   cf_path: ${exp.dir}/hydra/config.yaml
#   selection_criterion: "latest" #"best_valacc" #best_valacc # model selection criterion or "latest"
#   #  selection_mode: "l#max # model selection criterion or "latest"
#   best_ckpt_path: ${exp.version_dir}/${test.selection_criterion}.ckpt # latest or best
#   only_latest_version: True # if false looks for best metrics across all versions in exp_dir. Turn to false if resumed training.
#   devries_repro_ood_split: False
#   assim_ood_norm_flag: False
#   iid_set_split: "devries" # all, devries


@dataclass
class DataConfig:
    pass


@dataclass
class Config:
    # data: DataConfig = DataConfig()
    data: Any = MISSING

    trainer: TrainerConfig = TrainerConfig()

    exp: ExperimentConfig = ExperimentConfig()
    model: ModelConfig = ModelConfig()
    eval: Any = MISSING
    test: Any = MISSING


def init():
    store = ConfigStore.instance()
    store.store(name="config_schema", node=Config)
    store.store(group="data", name="data_schema", node=DataConfig)
