import subprocess
from datetime import datetime
from typing import Any

import pytest

from fd_shifts.experiments import get_all_experiments
from fd_shifts.experiments.launcher import BASH_BASE_COMMAND, BASH_LOCAL_COMMAND
from fd_shifts.tests.utils import mock_env_if_missing


def _update_overrides_fast(overrides: dict[str, Any]) -> dict[str, Any]:
    # HACK: This is highly machine dependend!
    max_batch_size = 16

    overrides["trainer.fast_dev_run"] = 5
    accum = overrides.get("trainer.batch_size", 128) // max_batch_size
    overrides["trainer.batch_size"] = max_batch_size
    overrides["trainer.accumulate_grad_batches"] = accum

    # HACK: Have to disable these because they do not handle limited batches
    overrides["eval.query_studies.noise_study"] = []
    overrides["eval.query_studies.in_class_study"] = []
    overrides["eval.query_studies.new_class_study"] = []
    return overrides


@pytest.mark.skip(
    "TODO: does nothing, remove or improve, also not compatible with new configs yet"
)
@pytest.mark.slow
def test_small_heuristic_run(mock_env_if_missing):
    # TODO: Test multiple with fixture
    name = "fd-shifts/cifar100_paper_sweep/confidnet_bbvgg13_do0_run1_rew2.2"
    # TODO: Also run some form of inference. Maybe generate outputs on main branch instead of using full experiments?
    mode = "train_test"

    experiments = list(
        filter(
            lambda experiment: str(experiment.to_path()) == name,
            get_all_experiments(),
        )
    )
    assert len(experiments) == 1
    experiment = experiments[0]

    log_file_name = (
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-"
        f"{str(experiment.to_path()).replace('/', '_').replace('.','_')}"
    )

    overrides = _update_overrides_fast(experiment.overrides())

    cmd = BASH_BASE_COMMAND.format(
        overrides=" ".join(f"{k}={v}" for k, v in overrides.items()),
        mode=mode,
    ).strip()

    cmd = BASH_LOCAL_COMMAND.format(command=cmd, log_file_name=log_file_name).strip()

    process = subprocess.run(cmd, shell=True, check=True)

    # TODO: check outputs against known-good
