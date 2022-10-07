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

BASH_LOCAL_COMMAND = r"""
bash -c 'set -o pipefail; {command} |& tee -a "/home/t974t/logs/{log_file_name}.log"'
"""

BASH_BASE_COMMAND = r"""
source .direnv/python-3.10.5/bin/activate &&
source .env &&
# source .envrc &&
python -W ignore fd_shifts/exec.py --config-path=$EXPERIMENT_ROOT_DIR/{config_path}/hydra/ --config-name=config exp.mode={mode} trainer.batch_size={batch_size}
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


async def worker(name, queue: asyncio.Queue[str], progress: Progress, task_id):
    while True:
        # Get a "work item" out of the queue.
        cmd = await queue.get()
        logger.info(f"{name} running {cmd}")
        proc = await asyncio.create_subprocess_shell(
            cmd,  # stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
        )

        # Wait for the subprocess exit.
        await proc.wait()

        if proc.returncode != 0:
            logger.error(f"{name} running {cmd} finished abnormally")
        else:
            logger.info(f"{name} running {cmd} finished")

        # data = await proc.stdout.read()
        # print(data.decode("utf-8"))

        # Notify the queue that the "work item" has been processed.
        progress.advance(task_id)
        queue.task_done()


async def run(_experiments: list[ValidationResult], mode: str):
    if len(_experiments) == 0:
        print("Nothing to run")
        return

    # Create a queue that we will use to store our "workload".
    queue: asyncio.Queue[str] = asyncio.Queue()

    for experiment in _experiments:
        log_file_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{str(experiment.experiment.to_path()).replace('/', '_').replace('.','_')}"

        cmd = BASH_BASE_COMMAND.format(
            config_path=experiment.experiment.to_path().relative_to("fd-shifts"),
            batch_size=get_batch_size(
                experiment.experiment.dataset, experiment.experiment.model, mode
            ),
            mode=mode,
        ).strip()

        cmd = BASH_LOCAL_COMMAND.format(command=cmd, log_file_name=log_file_name)

        # cmd = f"echo '{log_file_name}'; sleep 1"

        queue.put_nowait(cmd)

    with Progress() as progress:
        progress_task_id = progress.add_task("Test", total=len(_experiments))

        tasks = []
        # TODO: Flag for n_workers
        for i in range(4):
            task = asyncio.create_task(
                worker(f"worker-{i}", queue, progress, progress_task_id)
            )
            tasks.append(task)

        # Wait until the queue is fully processed.
        await queue.join()

        # Cancel our worker tasks.
        for task in tasks:
            task.cancel()
        # Wait until all worker tasks are cancelled.
        await asyncio.gather(*tasks, return_exceptions=True)


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
    local: bool,
    ignore_running: bool,
    jobs_list: list[str] | None,
):
    # if validation_file is not None:
    _experiments = parse_validation_file(validation_file)
    # else:
    #     _experiments = experiments.get_all_experiments()
    if jobs_list is not None:
        _experiments = list(
            filter(
                lambda e: str(e.experiment.to_path().relative_to("fd-shifts"))
                in jobs_list,
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
            if local:
                run(_experiments, mode)
            else:
                submit(_experiments, mode)

        return

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

    if not ignore_running:
        jobs = get_jobs()

        _experiments = list(
            filter(lambda e: not is_experiment_running(e, jobs), _experiments)
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
        if local:
            # run(_experiments, mode)
            asyncio.run(run(_experiments, mode))
        else:
            submit(_experiments, mode)


def add_arguments(parser: argparse.ArgumentParser):
    sub_parsers = parser.add_subparsers()

    parser.add_argument("--validation-file", default=None, type=Path)
    parser.add_argument("--config-exists", action="store_true")
    parser.add_argument("--model-exists", action="store_true")
    parser.add_argument("--config-valid", action="store_true")
    parser.add_argument("--outputs-valid", action="store_true")
    parser.add_argument("--results-valid", action="store_true")

    parser.add_argument("--study", default=None, type=str)
    parser.add_argument("--dataset", default=None, type=str)
    parser.add_argument("--dropout", default=None, type=int, choices=(0, 1))
    parser.add_argument(
        "--model", default=None, type=str, choices=("vit", "dg", "devries", "confidnet")
    )
    parser.add_argument(
        "--exclude-model",
        default=None,
        type=str,
        choices=("vit", "dg", "devries", "confidnet"),
    )
    parser.add_argument("--precision-study", action="store_true")

    parser.add_argument("-n", "--limit", default=None, type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--mode", default="test", choices=("test", "train", "train_test", "analysis")
    )
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--ignore-running", action="store_true")
    parser.add_argument("--jobs-list", default=None, type=Path)

    return parser


def main(args):
    jobs_list: list[str] | None = None

    if args.jobs_list is not None:
        with open(args.jobs_list, "rt") as f:
            jobs_list = f.read().split("\n")

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
        local=args.local,
        ignore_running=args.ignore_running,
        jobs_list=jobs_list,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args()

    main(args)
