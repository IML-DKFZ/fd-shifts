import argparse
import json
import sys
from pathlib import Path

from pssh.clients import SSHClient
from pssh.exceptions import Timeout
from rich.pretty import pprint

from fd_shifts import experiments
from fd_shifts.experiments.sync import sync_to_remote
from fd_shifts.experiments.validation import ValidationResult

# lang: bash
BSUB_COMMAND = r"""
bsub -gpu num=1:j_exclusive=yes:gmem=10.7G \
    -L /bin/bash \
    -q gpu-lowprio \
    -u 'till.bungert@dkfz-heidelberg.de' \
    -B \
    -m 'e230-dgxa100-1 e230-dgxa100-2 e230-dgx1-1' \
    -g /t974t/test \
    -J "{name}" \
    bash -li -c 'set -o pipefail; echo $LSB_JOBID && {command} |& tee -a "/home/t974t/logs/$LSB_JOBID.log"'
"""

# lang: bash
BASE_COMMAND = r"""
source .envrc &&
python -W ignore fd_shifts/exec.py --config-path=$EXPERIMENT_ROOT_DIR/{config_path}/hydra/ --config-name=config exp.mode={mode} trainer.batch_size={batch_size}
"""


def get_batch_size(dataset: str, model: str):
    if model == "vit":
        return 80

    if dataset in ["wilds_animals", "animals", "breeds"]:
        return 128

    return 512


def parse_validation_file(validation_file: Path) -> list[ValidationResult]:
    with validation_file.open() as file:
        _experiments = json.load(file)

    _experiments = list(map(lambda t: ValidationResult(**t[1]), _experiments.items()))
    for exp in _experiments:
        exp.experiment = experiments.Experiment(**exp.experiment)
        exp.logs = []
    return _experiments


def submit(_experiments: list[ValidationResult], mode: str):
    if len(_experiments) == 0:
        print("Nothing to run")
        return

    client = SSHClient("odcf-worker01.inet.dkfz-heidelberg.de")

    for experiment in _experiments:
        sync_to_remote(experiment.experiment.to_path())

        cmd = BASE_COMMAND.format(
            config_path=experiment.experiment.to_path().relative_to("fd-shifts"),
            batch_size=get_batch_size(
                experiment.experiment.dataset, experiment.experiment.model
            ),
            mode=mode,
        ).strip()
        cmd = BSUB_COMMAND.format(
            name=experiment.experiment.to_path().relative_to("fd-shifts"), command=cmd
        ).strip()

        with client.open_shell(read_timeout=1) as shell:
            shell.run("cd failure-detection-benchmark")
            shell.run("source .envrc")
            shell.run(cmd)

            try:
                for line in shell.stdout:
                    pprint(line)
            except Timeout:
                pass

            try:
                for line in shell.stderr:
                    pprint(line)
            except Timeout:
                pass

        for line in shell.stdout:
            pprint(line)

        for line in shell.stderr:
            pprint(line)


def launch(
    validation_file: Path | None,
    study: str | None,
    dataset: str | None,
    dropout: int | None,
    backbone: str | None,
    exclude_backbone: str | None,
    model_exists: bool,
    config_exists: bool,
    config_valid: bool,
    outputs_valid: bool,
    results_valid: bool,
    mode: str,
    dry_run: bool,
    n: int | None,
    precision_study: bool,
):
    # if validation_file is not None:
    _experiments = parse_validation_file(validation_file)
    # else:
    #     _experiments = experiments.get_all_experiments()

    _experiments = list(
        filter(
            lambda experiment: experiment.config_exists == config_exists,
            _experiments,
        )
    )

    _experiments = list(
        filter(
            lambda experiment: experiment.model_exists == model_exists,
            _experiments,
        )
    )

    _experiments = list(
        filter(
            lambda experiment: experiment.config_valid == config_valid,
            _experiments,
        )
    )

    _experiments = list(
        filter(
            lambda experiment: experiment.outputs_valid == outputs_valid,
            _experiments,
        )
    )

    _experiments = list(
        filter(
            lambda experiment: experiment.results_valid == results_valid,
            _experiments,
        )
    )

    if dataset is not None:
        _experiments = list(
            filter(
                lambda experiment: experiment.experiment.dataset == dataset,
                _experiments,
            )
        )

    if dropout is not None:
        _experiments = list(
            filter(
                lambda experiment: experiment.experiment.dropout == dropout,
                _experiments,
            )
        )

    if backbone is not None:
        _experiments = list(
            filter(
                lambda experiment: experiment.experiment.model == backbone,
                _experiments,
            )
        )

    if exclude_backbone is not None:
        _experiments = list(
            filter(
                lambda experiment: experiment.experiment.model != exclude_backbone,
                _experiments,
            )
        )

    if n is not None:
        _experiments = _experiments[:n]

    if precision_study:
        _experiments = list(
            filter(
                lambda experiment: "precision_study"
                in str(experiment.experiment.group_dir)
                and "64" not in str(experiment.experiment.group_dir),
                _experiments,
            )
        )
    else:
        _experiments = list(
            filter(
                lambda experiment: "precision_study"
                not in str(experiment.experiment.group_dir),
                _experiments,
            )
        )

    pprint(
        list(
            map(
                lambda exp: exp.experiment.to_path(),
                _experiments,
            )
        )
    )

    if not dry_run:
        submit(_experiments, mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-file", default=None, type=Path)
    parser.add_argument("--study", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--dropout", default=None, type=int)
    parser.add_argument("--model", default=None)
    parser.add_argument("--exclude-model", default=None)
    parser.add_argument("-n", "--limit", default=None, type=int)
    parser.add_argument("--config-exists", action="store_true")
    parser.add_argument("--model-exists", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config-valid", action="store_true")
    parser.add_argument("--outputs-valid", action="store_true")
    parser.add_argument("--results-valid", action="store_true")
    parser.add_argument("--precision-study", action="store_true")
    parser.add_argument("--mode", default="test")

    args = parser.parse_args()

    launch(
        validation_file=args.validation_file,
        study=args.study,
        dataset=args.dataset,
        dropout=args.dropout,
        backbone=args.model,
        exclude_backbone=args.exclude_model,
        config_exists=args.config_exists,
        model_exists=args.model_exists,
        config_valid=args.config_valid,
        outputs_valid=args.outputs_valid,
        results_valid=args.results_valid,
        mode=args.mode,
        dry_run=args.dry_run,
        n=args.limit,
        precision_study=args.precision_study,
    )


if __name__ == "__main__":
    main()
