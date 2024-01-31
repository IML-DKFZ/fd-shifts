import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from fd_shifts.experiments import get_all_experiments
from fd_shifts.experiments.launcher import BASH_BASE_COMMAND, BASH_LOCAL_COMMAND
from fd_shifts.tests.utils import mock_env_if_missing


def _update_overrides_fast(overrides: dict[str, Any]) -> dict[str, Any]:
    # HACK: This is highly machine dependend!
    max_batch_size = 4

    overrides["trainer.fast_dev_run"] = 5
    accum = overrides.get("trainer.batch_size", 128) // max_batch_size
    overrides["trainer.batch_size"] = max_batch_size
    overrides["trainer.accumulate_grad_batches"] = accum

    return overrides


@pytest.mark.skip("TODO: not compatible with new configs yet")
@pytest.mark.slow
@pytest.mark.parametrize(
    "exp_name",
    [
        "fd-shifts/cifar10_paper_sweep/confidnet_bbvgg13_do0_run1_rew2.2",
        "fd-shifts/vit/svhn_modeldg_bbvit_lr0.01_bs128_run4_do1_rew10",
        "fd-shifts/vit/svhn_modelvit_bbvit_lr0.03_bs128_run0_do1_rew0",
        "medshifts/ms_dermoscopyall/confidnet_bbefficientnetb4_run1",
        "medshifts/ms_dermoscopyallbutbarcelona/confidnet_bbefficientnetb4_run1",
        "medshifts/ms_dermoscopyallbutmskcc/deepgamblers_bbefficientnetb4_run1",
        "medshifts/ms_dermoscopyallham10000multi/devries_bbefficientnetb4_run1",
        "medshifts/ms_dermoscopyallham10000subclass/confidnet_bbefficientnetb4_run1",
        "medshifts/ms_lidc_idriall/confidnet_bbdensenet121_run1",
        "medshifts/ms_lidc_idriall_calcification/confidnet_bbdensenet121_run1",
        "medshifts/ms_lidc_idriall_spiculation/deepgamblers_bbdensenet121_run1",
        "medshifts/ms_lidc_idriall_texture/devries_bbdensenet121_run1",
        "medshifts/ms_rxrx1all/confidnet_bbdensenet161_run1",
        "medshifts/ms_rxrx1all_large_set1/devries_bbdensenet161_run1",
        "medshifts/ms_rxrx1all_large_set2/deepgamblers_bbdensenet161_run1",
        "medshifts/ms_xray_chestall/confidnet_bbdensenet121_run1",
        "medshifts/ms_xray_chestallbutchexpert/deepgamblers_bbdensenet121_run1",
        "medshifts/ms_xray_chestallbutnih14/devries_bbdensenet121_run1",
    ],
)
def test_run_experiment(exp_name: str):
    experiments = list(
        filter(
            lambda experiment: str(experiment.to_path()) == exp_name,
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
        mode="train_test",
    ).strip()

    cmd = BASH_LOCAL_COMMAND.format(command=cmd, log_file_name=log_file_name).strip()

    subprocess.run(cmd, shell=True, check=True)

    exp_path = experiment.to_path()

    if exp_path.is_relative_to("medshifts"):
        exp_path = exp_path.relative_to("medshifts")
    if exp_path.is_relative_to("fd-shifts"):
        exp_path = exp_path.relative_to("fd-shifts")

    exp_path = (
        Path(os.getenv("EXPERIMENT_ROOT_DIR", default="./experiments")) / exp_path
    )

    assert exp_path.is_dir()

    # check model checkpoint
    assert list(exp_path.glob("version*")) != []
    assert list(exp_path.glob("version*/last.ckpt")) != []

    # check test outputs
    assert (exp_path / "test_results").is_dir()
    test_output_files = [
        "analysis_metrics_iid_study.csv",
        "analysis_metrics_val_tuning.csv",
        "external_confids.npz",
        # "external_confids_dist.npz",
        "raw_logits.npz",
        # "raw_logits_dist.npz",
    ]
    assert all(
        (exp_path / "test_results" / f).is_file() for f in test_output_files
    ), f"Expected test outputs not found: {list(filter(lambda f: not (exp_path / 'test_results' / f).is_file(), test_output_files))}"

    assert all(
        os.path.getsize(exp_path / "test_results" / f) > 0 for f in test_output_files
    ), f"Expected test outputs empty: {list(filter(lambda f: os.path.getsize(exp_path / 'test_results' / f) <= 0, test_output_files))}"
