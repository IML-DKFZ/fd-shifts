import collections.abc
import json
import os
import subprocess
from pathlib import Path

import pytest
from deepdiff import DeepDiff
from omegaconf import DictConfig, OmegaConf
from rich import print

from fd_shifts import configs


@pytest.fixture
def mock_env_if_missing(monkeypatch) -> None:
    monkeypatch.setenv(
        "EXPERIMENT_ROOT_DIR", os.getenv("EXPERIMENT_ROOT_DIR", default="./experiments")
    )
    monkeypatch.setenv(
        "DATASET_ROOT_DIR", os.getenv("DATASET_ROOT_DIR", default="./data")
    )


def _extract_config_from_cli_stderr(output: str) -> str:
    _, _, config_yaml = output.partition("BEGIN CONFIG")
    config_yaml, _, _ = config_yaml.partition("END CONFIG")

    return config_yaml  # .strip()


def to_dict(obj):
    return json.loads(json.dumps(obj, default=lambda o: getattr(o, "__dict__", str(o))))


def test_api_and_main_same(mock_env_if_missing) -> None:
    study = "deepgamblers"
    data = "svhn"

    output: str = subprocess.run(
        f"python fd_shifts/exec.py study={study} data={data}_data exp.mode=debug",
        shell=True,
        capture_output=True,
        check=True,
    ).stderr.decode("utf-8")

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
