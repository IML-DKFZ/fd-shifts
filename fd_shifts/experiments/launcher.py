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
    -N \
    -g /t974t/test \
    -J "_test" \
    '{command}'
"""

# lang: bash
BASE_COMMAND = r"""
source ~/.env &&
source /dkfz/cluster/gpu/data/OE0612/t974t/venv/fd-shifts/bin/activate &&
python -W ignore fd_shifts/exec.py config={config_path}
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

    pprint(_experiments)

    cmd = BASE_COMMAND.format(config_path=_experiments[0].experiment.to_path()).strip()
    cmd = BSUB_COMMAND.format(command=cmd).strip()


    client = SSHClient("odcf-worker01.inet.dkfz-heidelberg.de")

    with client.open_shell(read_timeout=1) as shell:
        shell.run("cd failure-detection-benchmark")
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-file", default=None, type=Path)
    parser.add_argument("--study", default=None)
    parser.add_argument("--dataset", default=None)

    args = parser.parse_args()

    launch(args.validation_file, args.study, args.dataset)


if __name__ == "__main__":
    main()
