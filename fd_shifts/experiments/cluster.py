import argparse
import asyncio
import json
import subprocess
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

from pssh.clients import SSHClient
from pssh.exceptions import Timeout
from rich.pretty import pprint
from rich.progress import Progress

from fd_shifts import experiments, logger
from fd_shifts.experiments.sync import sync_to_remote
from fd_shifts.experiments.validation import ValidationResult

BASH_BSUB_COMMAND = r"""
bsub -gpu num=1:j_exclusive=yes:gmem={gmem}\
    -L /bin/bash \
    -q gpu \
    -u 'till.bungert@dkfz-heidelberg.de' \
    -B {nodes} \
    -R "select[hname!='e230-dgx2-2']" \
    -g /t974t/test \
    -J "{name}" \
    bash -li -c 'set -o pipefail; echo $LSB_JOBID && {command} |& tee -a "/home/t974t/logs/$LSB_JOBID.log"'
"""


def get_jobs() -> list[dict[str, str]]:
    with urllib.request.urlopen("http://localhost:3030/jobs") as response:
        records: list[dict[str, str]] = json.loads(response.read())["RECORDS"]
    return records


def is_experiment_running(
    experiment: ValidationResult, jobs: list[dict[str, str]]
) -> bool:
    _experiments = list(
        map(
            lambda j: j["JOB_NAME"],
            filter(lambda j: j["STAT"] in ("RUN", "PEND", "DONE"), jobs),
        )
    )
    running = (
        str(experiment.experiment.to_path().relative_to("fd-shifts")) in _experiments
    )

    if running:
        print(f"{experiment.experiment.to_path()} is already running")

    return running


def get_batch_size(dataset: str, model: str, mode: str):
    match mode:
        case "test":
            # if model == "vit":
            #     return 80
            #
            # if dataset in ["wilds_animals", "animals", "breeds"]:
            #     return 128
            #
            # return 512
            if model == "vit":
                return 80

            if dataset in [
                "wilds_animals",
                "animals",
                "wilds_animals_openset",
                "animals_openset",
                "breeds",
            ]:
                return 128

            return 512
        case "train" | "train_test":
            match model:
                case "vit":
                    match dataset:
                        case "svhn" | "svhn_openset" | "cifar10" | "breeds":
                            return "64 +trainer.accumulate_grad_batches=2"
                        case "wilds_animals" | "wilds_animals_openset" | "cifar100" | "super_cifar100":
                            return "64 +trainer.accumulate_grad_batches=8"
                case _:
                    match dataset:
                        case "svhn" | "svhn_openset" | "cifar10" | "cifar100" | "super_cifar100" | "breeds":
                            return 128
                        case "wilds_animals" | "wilds_animals_openset" | "animals" | "animals_openset":
                            return 16
                        case "wilds_camelyon":
                            return 32
        case _:
            return 128


def get_nodes(mode: str):
    match mode:
        case "train" | "train_test":
            return "-sp 36"
        case _:
            return ""


def get_gmem(mode: str, model: str):
    match mode:
        case "train" | "train_test":
            match model:
                case "vit":
                    return "33G"
                case _:
                    # return "10.7G"
                    return "33G"
        case _:
            match model:
                case "vit":
                    return "33G"
                case _:
                    # return "10.7G"
                    return "33G"


def submit(_experiments: list[ValidationResult], mode: str):
    if len(_experiments) == 0:
        print("Nothing to run")
        return

    client = SSHClient("odcf-worker01.inet.dkfz-heidelberg.de")

    for experiment in _experiments:
        sync_to_remote(experiment.experiment.to_path())

        cmd = BASH_BASE_COMMAND.format(
            config_path=experiment.experiment.to_path().relative_to("fd-shifts"),
            batch_size=get_batch_size(
                experiment.experiment.dataset, experiment.experiment.model, mode
            ),
            mode=mode,
        ).strip()
        cmd = BASH_BSUB_COMMAND.format(
            name=experiment.experiment.to_path().relative_to("fd-shifts"),
            command=cmd,
            nodes=get_nodes(mode),
            gmem=get_gmem(mode, experiment.experiment.model),
        ).strip()

        print(cmd)

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
