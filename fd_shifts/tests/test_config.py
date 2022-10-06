import json
import os
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path
from typing import Any, Optional, cast

import hydra
import pytest
from deepdiff import DeepDiff
from hydra import compose, initialize_config_module
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra.errors import ConfigCompositionException
from omegaconf import DictConfig, ListConfig, OmegaConf
from rich.pretty import pprint

from fd_shifts import configs
from fd_shifts.experiments import Experiment, get_all_experiments


@pytest.fixture
def mock_env_if_missing(monkeypatch) -> None:
    monkeypatch.setenv(
        "EXPERIMENT_ROOT_DIR", os.getenv("EXPERIMENT_ROOT_DIR", default="./experiments")
    )
    monkeypatch.setenv(
        "DATASET_ROOT_DIR", os.getenv("DATASET_ROOT_DIR", default="./data")
    )


def initialize_hydra(overrides: list[str]) -> DictConfig:
    """Takes the place of hydra.main"""
    default_overrides = [
        "hydra.job.num=0",
        "hydra.job.id=0",
        "hydra.runtime.output_dir=.",
        "hydra.hydra_help.hydra_help=''",
    ]

    with initialize_config_module(version_base=None, config_module="fd_shifts.configs"):
        cfg = compose(
            "config", overrides=overrides + default_overrides, return_hydra_config=True
        )
        HydraConfig.instance().set_config(cfg)

    return cfg


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ([], nullcontext()),
        (["exp.log_path=test"], nullcontext()),
        (["exp.global_seed=1234"], nullcontext()),
        (["pkgversion=0.0.1+aefe5c8"], pytest.raises(ValueError)),
        (["trainer.val_split=foo"], pytest.raises(ConfigCompositionException)),
        (["eval.query_studies.iid_study=svhn"], nullcontext()),
        (
            ["eval.query_studies.iid_study=doesnt_exist_dataset"],
            pytest.raises(ValueError),
        ),
    ],
)
def test_validation(
    overrides: list[str], expected: AbstractContextManager, mock_env_if_missing: Any
):
    with expected:
        configs.init()

        dcfg = initialize_hydra(overrides)

        cfg: configs.Config = OmegaConf.to_object(dcfg)
        cfg.validate()
        pprint(OmegaConf.to_yaml(cfg, resolve=True))
        assert cfg.exp.root_dir == Path(os.getenv("EXPERIMENT_ROOT_DIR", default=""))


@pytest.mark.parametrize(
    ("study",),
    [
        ("vit",),
        ("confidnet",),
        ("deepgamblers",),
        ("devries",),
    ],
)
def test_existing_studies(study: str, mock_env_if_missing: Any):
    configs.init()
    overrides = [f"study={study}"]

    dcfg = initialize_hydra(overrides)

    print(type(dcfg))
    cfg: configs.Config = OmegaConf.to_object(dcfg)
    pprint(OmegaConf.to_yaml(cfg, resolve=False))
    print(type(cfg))


def _normalize_dataset_name(dataset: str):
    return dataset.replace("_data", "").replace("_384", "").replace("_ood_test", "")


@pytest.mark.parametrize(
    ("dataset",),
    [
        ("breeds_384_data",),
        ("breeds_data",),
        ("breeds_ood_test_384_data",),
        ("breeds_ood_test_data",),
        ("cifar100_384_data",),
        ("cifar100_data",),
        ("cifar10_384_data",),
        ("cifar10_data",),
        ("corrupt_cifar100_384_data",),
        ("corrupt_cifar100_data",),
        ("corrupt_cifar10_384_data",),
        ("corrupt_cifar10_data",),
        ("super_cifar100_384_data",),
        ("super_cifar100_data",),
        ("svhn_384_data",),
        ("svhn_data",),
        ("svhn_openset_384_data",),
        ("svhn_openset_data",),
        ("tinyimagenet_384_data",),
        ("tinyimagenet_resize_data",),
        ("wilds_animals_384_data",),
        ("wilds_animals_data",),
        ("wilds_animals_ood_test_384_data",),
        ("wilds_animals_ood_test_data",),
        ("wilds_animals_openset_384_data",),
        ("wilds_animals_openset_data",),
        ("wilds_camelyon_384_data",),
        ("wilds_camelyon_data",),
        ("wilds_camelyon_ood_test_384_data",),
        ("wilds_camelyon_ood_test_data",),
    ],
)
def test_existing_datasets(dataset: str, mock_env_if_missing: Any):
    configs.init()
    overrides = [f"data={dataset}"]

    dcfg = initialize_hydra(overrides)

    print(type(dcfg))
    cfg: configs.Config = cast(
        configs.Config, OmegaConf.to_object(dcfg)
    )  # only affects the linter
    pprint(OmegaConf.to_yaml(cfg, resolve=False))
    assert isinstance(cfg, configs.Config)  # runtime type check
    assert cfg.data.dataset == _normalize_dataset_name(dataset)


@pytest.mark.parametrize(
    "experiment",
    list(
        filter(
            lambda experiment: not (experiment.model != "vit" and experiment.backbone == "vit")
            and "precision" not in str(experiment.group_dir),
            get_all_experiments(),
        )
    ),
)
def test_experiment_configs(experiment: Experiment):
    configs.init()

    base_path = Path("/media/experiments/")
    path = base_path / experiment.to_path()
    config_path = path / "hydra" / "config.yaml"

    dconf = OmegaConf.load(config_path)
    dconf._metadata.object_type = configs.Config

    def fix_metadata(cfg: DictConfig | ListConfig):
        if hasattr(cfg, "_target_"):
            cfg._metadata.object_type = getattr(configs, cfg._target_.split(".")[-1])
        for k, v in cfg.items():
            match v:
                case DictConfig():
                    fix_metadata(v)
                case _:
                    pass

    fix_metadata(dconf)

    schema = initialize_hydra(
        [f"{key}={value}" for key, value in experiment.overrides().items()]
    )
    oschema: configs.Config = OmegaConf.to_object(schema)  # type: ignore

    conf: configs.Config = OmegaConf.to_object(dconf)  # type: ignore
    conf.validate()

    def to_dict(obj):
        return json.loads(
            json.dumps(obj, default=lambda o: getattr(o, "__dict__", str(o)))
        )

    exclude_paths = {
        "root['hydra']",
        "root['data']['num_workers']",
        "root['data']['data_dir']",
        "root['exp']['dir']",
        "root['exp']['global_seed']",
        "root['exp']['group_dir']",
        # "root['exp']['group_name']",
        "root['exp']['log_path']",
        "root['exp']['mode']",
        # "root['exp']['name']",
        "root['exp']['output_paths']",
        "root['exp']['version']",
        "root['exp']['version_dir']",
        "root['exp']['crossval_ids_path']",
        "root['test']['best_ckpt_path']",
        "root['test']['cf_path']",
        "root['test']['dir']",
        "root['test']['external_confids_output_path']",
        "root['test']['raw_output_path']",
        "root['pkgversion']",
        "root['trainer']['val_every_n_epoch']",
        "root['eval']['confidence_measures']['train']",
        "root['eval']['confidence_measures']['val']",
        "root['eval']['confid_metrics']['train']",
        "root['data']['kwargs']['out_classes']",
        "root['trainer']['do_val']",
        # Remove this
        "root['model']['network']['imagenet_weights_path']",
        "root['trainer']['resume_from_ckpt_confidnet']",
        "root['model']['network']['save_dg_backbone_path']",
        "root['model']['avg_pool']",
    }

    if experiment.model != "dg":
        exclude_paths.add("root['trainer']['dg_pretrain_epochs']")
        exclude_paths.add("root['model']['dg_reward']")
        exclude_paths.add("root['model']['network']['save_dg_backbone_path']")

    if experiment.model == "vit":
        exclude_paths.add("root['trainer']['num_steps']")
        exclude_paths.add("root['trainer']['lr_scheduler']['max_epochs']")

    if experiment.model != "confidnet":
        exclude_paths.add("root['trainer']['resume_from_ckpt_confidnet']")
        exclude_paths.add("root['trainer']['num_epochs_backbone']")
        exclude_paths.add("root['model']['confidnet_fc_dim']")

    if conf.trainer.optimizer._target_ == "torch.optim.SGD":
        conf.trainer.optimizer._target_ = "torch.optim.sgd.SGD"
        conf.trainer.optimizer._partial_ = True

    if experiment.model != "vit" and experiment.dataset == "animals":
        conf.trainer.batch_size = 16
    if experiment.model != "vit" and experiment.dataset == "animals_openset":
        conf.trainer.batch_size = 16

    if experiment.model != "vit" and experiment.dataset == "camelyon":
        conf.trainer.batch_size = 32

    config_diff = DeepDiff(
        to_dict(oschema),
        to_dict(conf),
        ignore_order=True,
        ignore_numeric_type_changes=True,
        exclude_paths=exclude_paths,
    )

    if config_diff:
        pprint(config_diff)
        raise AssertionError("Configs do not match")
