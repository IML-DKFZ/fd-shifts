import collections.abc
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

import pytest
import rich
import yaml
from deepdiff import DeepDiff
from omegaconf import DictConfig, OmegaConf
from rich import print

from fd_shifts import configs
from fd_shifts.experiments import get_ms_experiments
from fd_shifts.experiments.launcher import BASH_BASE_COMMAND, BASH_LOCAL_COMMAND
from fd_shifts.tests.utils import mock_env_if_missing


def _extract_config_from_cli_stderr(output: str) -> str:
    _, _, config_yaml = output.partition("BEGIN CONFIG")
    config_yaml, _, _ = config_yaml.partition("END CONFIG")

    return config_yaml


def to_dict(obj):
    return json.loads(json.dumps(obj, default=lambda o: getattr(o, "__dict__", str(o))))


@pytest.mark.skip("TODO: not compatible with new configs yet")
def test_api_and_main_same(mock_env_if_missing) -> None:
    study = "deepgamblers"
    data = "svhn"

    output: str = subprocess.run(
        f"python fd_shifts/exec.py study={study} data={data}_data exp.mode=debug",
        shell=True,
        capture_output=True,
        check=False,
    ).stderr.decode("utf-8")

    print(output)

    config_yaml = _extract_config_from_cli_stderr(output)
    config_cli = OmegaConf.to_container(OmegaConf.create(config_yaml))
    config_cli = configs.Config(**config_cli)
    config_cli.__pydantic_validate_values__()

    configs.init()

    config_api = configs.Config.with_defaults(study, data)

    diff = DeepDiff(
        to_dict(config_api),
        to_dict(config_cli),
        ignore_order=True,
        exclude_paths=["root['exp']['global_seed']"],
    )

    assert not diff


ms_experiments = {str(exp.to_path()): exp for exp in get_ms_experiments()}


@pytest.mark.skip("TODO: not compatible with new configs yet")
@pytest.mark.slow
@pytest.mark.parametrize(
    "exp_name",
    [
        "medshifts/ms_dermoscopyall/confidnet_bbefficientnetb4_run1",
        "medshifts/ms_dermoscopyall/deepgamblers_bbefficientnetb4_run1",
        "medshifts/ms_dermoscopyall/devries_bbefficientnetb4_run1",
        "medshifts/ms_dermoscopyallbutbarcelona/confidnet_bbefficientnetb4_run1",
        "medshifts/ms_dermoscopyallbutbarcelona/deepgamblers_bbefficientnetb4_run1",
        "medshifts/ms_dermoscopyallbutbarcelona/devries_bbefficientnetb4_run1",
        "medshifts/ms_dermoscopyallbutmskcc/confidnet_bbefficientnetb4_run1",
        "medshifts/ms_dermoscopyallbutmskcc/deepgamblers_bbefficientnetb4_run1",
        "medshifts/ms_dermoscopyallbutmskcc/devries_bbefficientnetb4_run1",
        "medshifts/ms_dermoscopyallham10000multi/confidnet_bbefficientnetb4_run1",
        "medshifts/ms_dermoscopyallham10000multi/deepgamblers_bbefficientnetb4_run1",
        "medshifts/ms_dermoscopyallham10000multi/devries_bbefficientnetb4_run1",
        "medshifts/ms_dermoscopyallham10000subclass/confidnet_bbefficientnetb4_run1",
        "medshifts/ms_dermoscopyallham10000subclass/deepgamblers_bbefficientnetb4_run1",
        "medshifts/ms_dermoscopyallham10000subclass/devries_bbefficientnetb4_run1",
        "medshifts/ms_lidc_idriall/confidnet_bbdensenet121_run1",
        "medshifts/ms_lidc_idriall/deepgamblers_bbdensenet121_run1",
        "medshifts/ms_lidc_idriall/devries_bbdensenet121_run1",
        "medshifts/ms_lidc_idriall_calcification/confidnet_bbdensenet121_run1",
        "medshifts/ms_lidc_idriall_calcification/deepgamblers_bbdensenet121_run1",
        "medshifts/ms_lidc_idriall_calcification/devries_bbdensenet121_run1",
        "medshifts/ms_lidc_idriall_spiculation/confidnet_bbdensenet121_run1",
        "medshifts/ms_lidc_idriall_spiculation/deepgamblers_bbdensenet121_run1",
        "medshifts/ms_lidc_idriall_spiculation/devries_bbdensenet121_run1",
        "medshifts/ms_lidc_idriall_texture/confidnet_bbdensenet121_run1",
        "medshifts/ms_lidc_idriall_texture/deepgamblers_bbdensenet121_run1",
        "medshifts/ms_lidc_idriall_texture/devries_bbdensenet121_run1",
        "medshifts/ms_rxrx1all/confidnet_bbdensenet161_run1",
        "medshifts/ms_rxrx1all/deepgamblers_bbdensenet161_run1",
        "medshifts/ms_rxrx1all/devries_bbdensenet161_run1",
        "medshifts/ms_rxrx1all_large_set1/confidnet_bbdensenet161_run1",
        "medshifts/ms_rxrx1all_large_set1/deepgamblers_bbdensenet161_run1",
        "medshifts/ms_rxrx1all_large_set1/devries_bbdensenet161_run1",
        "medshifts/ms_rxrx1all_large_set2/confidnet_bbdensenet161_run1",
        "medshifts/ms_rxrx1all_large_set2/deepgamblers_bbdensenet161_run1",
        "medshifts/ms_rxrx1all_large_set2/devries_bbdensenet161_run1",
        "medshifts/ms_xray_chestall/confidnet_bbdensenet121_run1",
        "medshifts/ms_xray_chestall/deepgamblers_bbdensenet121_run1",
        "medshifts/ms_xray_chestall/devries_bbdensenet121_run1",
        "medshifts/ms_xray_chestallbutchexpert/confidnet_bbdensenet121_run1",
        "medshifts/ms_xray_chestallbutchexpert/deepgamblers_bbdensenet121_run1",
        "medshifts/ms_xray_chestallbutchexpert/devries_bbdensenet121_run1",
        "medshifts/ms_xray_chestallbutnih14/confidnet_bbdensenet121_run1",
        "medshifts/ms_xray_chestallbutnih14/deepgamblers_bbdensenet121_run1",
        "medshifts/ms_xray_chestallbutnih14/devries_bbdensenet121_run1",
    ],
)
def test_medshifts_experiment_configs(exp_name) -> None:
    experiment = ms_experiments[exp_name]

    log_file_name = (
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-"
        f"{str(experiment.to_path()).replace('/', '_').replace('.','_')}"
    )

    overrides = experiment.overrides()

    cmd = BASH_BASE_COMMAND.format(
        overrides=" ".join(f"{k}={v}" for k, v in overrides.items()),
        mode="debug",
    ).strip()

    cmd = BASH_LOCAL_COMMAND.format(command=cmd, log_file_name=log_file_name).strip()

    subprocess.run(cmd, shell=True, check=True)

    gen_config_path = (
        Path(os.getenv("EXPERIMENT_ROOT_DIR", default="."))
        / experiment.to_path().relative_to("medshifts")
        / "hydra"
        / "config.yaml"
    )

    print(gen_config_path)

    assert gen_config_path.exists()

    with gen_config_path.open("r") as file:
        generated_config = yaml.unsafe_load(file)

    epochs = generated_config["trainer"]["num_epochs"]
    if isinstance(epochs, str):
        match generated_config["eval"]["ext_confid_name"]:
            case "tcp":
                match = re.search(r"tcp,(\d+)", epochs)[1]
            case "dg":
                match = re.search(r"dg,(\d+)", epochs)[1]
            case _:
                match = re.search(r"\d+,(\d+)", epochs)[1]
        generated_config["trainer"]["num_epochs"] = int(match)
    # generated_config = OmegaConf.load(gen_config_path)
    # generated_config = OmegaConf.to_container(generated_config, resolve=False)

    base_path = Path(
        "/home/t974t/NetworkDrives/E130-Personal/Kobelke/cluster_checkpoints"
    )
    ref_path = f"ms_{experiment.dataset}_run_1/{experiment.model}_mcd/hydra/config.yaml"

    assert (base_path / ref_path).exists()
    with (base_path / ref_path).open("r") as file:
        reference_config = yaml.unsafe_load(file)

    reference_config["trainer"]["optimizer"]["lr"] = reference_config["trainer"][
        "optimizer"
    ]["learning_rate"]
    del reference_config["trainer"]["optimizer"]["learning_rate"]
    optim_name: str = reference_config["trainer"]["optimizer"]["name"]
    reference_config["trainer"]["optimizer"][
        "_target_"
    ] = f"torch.optim.{optim_name.lower()}.{optim_name.capitalize()}"
    del reference_config["trainer"]["optimizer"]["name"]

    lr_name = reference_config["trainer"]["lr_scheduler"]["name"]
    reference_config["trainer"]["lr_scheduler"][
        "_target_"
    ] = f"torch.optim.lr_scheduler.{lr_name}LR"
    del reference_config["trainer"]["lr_scheduler"]["name"]

    if "noise_study" not in reference_config["eval"]["query_studies"]:
        reference_config["eval"]["query_studies"]["noise_study"] = []
    if "in_class_study" not in reference_config["eval"]["query_studies"]:
        reference_config["eval"]["query_studies"]["in_class_study"] = []
    if "new_class_study" not in reference_config["eval"]["query_studies"]:
        reference_config["eval"]["query_studies"]["new_class_study"] = []

    exclude_paths = [
        "root['data']['data_dir']",
        "root['eval']['confidence_measures']['test']",
        "root['eval']['test_conf_scaling']",
        "root['exp']['data_root_dir']",
        "root['exp']['global_seed']",
        "root['exp']['group_dir']",
        "root['exp']['group_name']",
        "root['exp']['log_path']",
        "root['exp']['mode']",
        "root['exp']['name']",
        "root['exp']['output_paths']['fit']['attributions_output']",
        "root['exp']['output_paths']['fit']['encoded_output']",
        "root['exp']['output_paths']['test']['input_imgs_plot']",
        "root['exp']['output_paths']['test']['raw_output']",
        "root['exp']['output_paths']['test']['raw_output_dist']",
        "root['exp']['root_dir']",
        "root['exp']['version']",
        "root['model']['balanced_sampling']",
        "root['model']['monitor_mcd_samples']",
        "root['model']['name']",
        "root['model']['rotate_at_testtime']",
        "root['pkgversion']",
        "root['test']['external_confids_output_path']",
        "root['test']['output_precision']",
        "root['test']['raw_output_path']",
        "root['test']['selection_mode']",
        "root['trainer']['accelerator']",
        "root['trainer']['accumulate_grad_batches']",
        "root['trainer']['num_steps']",
        "root['trainer']['val_every_n_epoch']",
        "root['trainer']['optimizer']['betas']",
        "root['trainer']['optimizer']['eps']",
        "root['model']['network']['save_dg_backbone_path']",
        # check
        "root['trainer']['num_epochs_backbone']",
        "root['trainer']['optimizer']['_partial_']",
        "root['trainer']['optimizer']['dampening']",
        "root['trainer']['optimizer']['maximize']",
        "root['trainer']['momentum']",
        "root['trainer']['weight_decay']",
        "root['trainer']['learning_rate']",
        "root['trainer']['lr_scheduler']['_partial_']",
        "root['trainer']['lr_scheduler']['last_epoch']",
        "root['trainer']['lr_scheduler']['milestones']",
        "root['trainer']['lr_scheduler']['verbose']",
        "root['trainer']['lr_scheduler']['T_max']",
        "root['trainer']['lr_scheduler']['max_epochs']",
        "root['trainer']['lr_scheduler']['eta_min']",
        "root['trainer']['lr_scheduler']['gamma']",
        "root['trainer']['val_split']",
        "root['test']['iid_set_split']",
        "root['trainer']['callbacks']['monitor']",
    ]

    if optim_name == "ADAM":
        exclude_paths.extend(
            [
                "root['trainer']['optimizer']['nesterov']",
            ]
        )

    if experiment.model != "confidnet":
        exclude_paths.extend(
            [
                "root['trainer']['learning_rate_confidnet']",
                "root['trainer']['learning_rate_confidnet_finetune']",
                "root['model']['confidnet_fc_dim']",
            ]
        )
    if experiment.model != "deepgamblers":
        exclude_paths.extend(
            [
                "root['model']['dg_reward']",
                "root['trainer']['dg_pretrain_epochs']",
                "root['model']['budget']",
                "root['model']['network']['load_dg_backbone_path']",
            ]
        )

    diff = DeepDiff(
        reference_config,
        generated_config,
        ignore_order=True,
        ignore_numeric_type_changes=True,
        exclude_paths=exclude_paths,
    )

    if diff:
        rich.print(diff)
    assert not diff, "Generated config does not match reference config"
