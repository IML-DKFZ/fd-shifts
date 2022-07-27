import argparse
import json
import sys
from pathlib import Path

from pssh.clients import SSHClient
from pssh.exceptions import Timeout
from rich.pretty import pprint

from fd_shifts import experiments
from fd_shifts.experiments.validation import ValidationResult

# lang: bash
BSUB_COMMAND = r"""
bsub -gpu num=1:j_exclusive=yes:gmem=10.7G \
    -L /bin/bash \
    -q gpu-lowprio \
    -u 'till.bungert@dkfz-heidelberg.de' \
    -B \
    -g /t974t/test \
    -J "{name}" \
    bash -li -c 'echo $LSB_JOBID && {command} |& tee -a "/home/t974t/logs/$LSB_JOBID.log"'
"""

# lang: bash
BASE_COMMAND = r"""
source .envrc &&
python -W ignore fd_shifts/exec.py --config-path=$EXPERIMENT_ROOT_DIR/{config_path}/hydra/ --config-name=config exp.mode=test trainer.batch_size=64
"""


def parse_validation_file(validation_file: Path) -> list[ValidationResult]:
    with validation_file.open() as file:
        _experiments = json.load(file)

    _experiments = list(map(lambda t: ValidationResult(**t[1]), _experiments.items()))
    for exp in _experiments:
        exp.experiment = experiments.Experiment(**exp.experiment)
        exp.logs = []
    return _experiments


def launch(validation_file: Path | None, study: str | None, dataset: str | None):
    # if validation_file is not None:
    _experiments = parse_validation_file(validation_file)
    # else:
    #     _experiments = experiments.get_all_experiments()

    _experiments = list(
        filter(
            lambda experiment: experiment.exists,
            _experiments,
        )
    )

    _experiments = list(
        filter(
            lambda experiment: experiment.config_valid,
            _experiments,
        )
    )

    _experiments = list(
        filter(
            lambda experiment: not experiment.outputs_valid,
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

    pprint(_experiments[0].experiment.to_path().relative_to("fd-shifts"))
    # return

    cmd = BASE_COMMAND.format(
        config_path=_experiments[0].experiment.to_path().relative_to("fd-shifts")
    ).strip()
    cmd = BSUB_COMMAND.format(
        name=_experiments[0].experiment.to_path().relative_to("fd-shifts"), command=cmd
    ).strip()

    client = SSHClient("odcf-worker01.inet.dkfz-heidelberg.de")

    with client.open_shell(read_timeout=1) as shell:
        shell.run("cd failure-detection-benchmark")
        shell.run("source .envrc")
        shell.run(cmd)

        try:
            for line in shell.stdout:
                print(line)
        except Timeout:
            pass

        try:
            for line in shell.stderr:
                print(line, file=sys.stderr)
        except Timeout:
            pass

    for line in shell.stdout:
        print(line)

    for line in shell.stderr:
        print(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-file", default=None, type=Path)
    parser.add_argument("--study", default=None)
    parser.add_argument("--dataset", default=None)

    args = parser.parse_args()

    launch(args.validation_file, args.study, args.dataset)


if __name__ == "__main__":
    main()
