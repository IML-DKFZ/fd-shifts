import argparse
import asyncio
import json
import subprocess
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

import rich
from pssh.clients import SSHClient
from pssh.exceptions import Timeout
from rich.pretty import pprint
from rich.progress import Progress

from fd_shifts import experiments, logger
from fd_shifts.experiments.cluster import submit
from fd_shifts.experiments.sync import sync_to_remote
from fd_shifts.experiments.validation import ValidationResult

BASH_LOCAL_COMMAND = r"""
bash -c 'set -o pipefail; {command} |& tee -a "./logs/{log_file_name}.log"'
"""

# _fd_shifts_exec --config-path=$EXPERIMENT_ROOT_DIR/{config_path}/hydra/ --config-name=config exp.mode={mode} trainer.batch_size={batch_size}
BASH_BASE_COMMAND = r"""
_fd_shifts_exec {overrides} exp.mode={mode}
"""


def parse_validation_file(validation_file: Path) -> list[ValidationResult]:
    with validation_file.open() as file:
        _experiments = json.load(file)

    _experiments = list(map(lambda t: ValidationResult(**t[1]), _experiments.items()))
    for exp in _experiments:
        exp.experiment = experiments.Experiment(**exp.experiment)
        exp.logs = []
    return _experiments


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


async def run(_experiments: list[experiments.Experiment], mode: str):
    if len(_experiments) == 0:
        print("Nothing to run")
        return

    Path("./logs").mkdir(exist_ok=True)

    # Create a queue that we will use to store our "workload".
    queue: asyncio.Queue[str] = asyncio.Queue()

    for experiment in _experiments:
        log_file_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{str(experiment.to_path()).replace('/', '_').replace('.','_')}"

        cmd = BASH_BASE_COMMAND.format(
            overrides=" ".join(f"{k}={v}" for k, v in experiment.overrides().items()),
            mode=mode,
        ).strip()

        cmd = BASH_LOCAL_COMMAND.format(
            command=cmd, log_file_name=log_file_name
        ).strip()

        # cmd = f"echo '{log_file_name}'; sleep 1"

        queue.put_nowait(cmd)

    with Progress() as progress:
        progress_task_id = progress.add_task("Test", total=len(_experiments))

        tasks = []
        # TODO: Flag for n_workers
        for i in range(1):
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
    # validation_file: Path | None,
    # study: str | None,
    dataset: str | None,
    dropout: int | None,
    model: str | None,
    backbone: str | None,
    exclude_model: str | None,
    # model_exists: bool,
    # config_exists: bool,
    # config_valid: bool,
    # outputs_valid: bool,
    # results_valid: bool,
    mode: str,
    dry_run: bool,
    run_nr: int | None,
    rew: float | None,
    # precision_study: bool,
    # local: bool,
    cluster: bool,
    # ignore_running: bool,
    # jobs_list: list[str] | None,
    name: str | None,
):
    # if validation_file is not None:
    #     _experiments = parse_validation_file(validation_file)
    # else:
    _experiments = experiments.get_all_experiments()

    _experiments = list(
        filter(lambda e: "precision_study" not in str(e.to_path()), _experiments)
    )

    # HACK: Temporarily turn off special vit runs
    # _experiments = list(
    #     filter(lambda e: not (e.model != "vit" and e.backbone == "vit"), _experiments)
    # )

    # if jobs_list is not None:
    #     _experiments = list(
    #         filter(
    #             lambda e: str(e.experiment.to_path().relative_to("fd-shifts"))
    #             in jobs_list,
    #             _experiments,
    #         )
    #     )
    #     pprint(
    #         list(
    #             map(
    #                 lambda exp: exp.experiment.to_path(),
    #                 _experiments,
    #             )
    #         )
    #     )
    #
    #     if not dry_run:
    #         if local:
    #             run(_experiments, mode)
    #         else:
    #             submit(_experiments, mode)
    #
    #     return
    #
    # _experiments = list(
    #     filter(
    #         lambda experiment: experiment.config_exists == config_exists,
    #         _experiments,
    #     )
    # )
    #
    # _experiments = list(
    #     filter(
    #         lambda experiment: experiment.model_exists == model_exists,
    #         _experiments,
    #     )
    # )
    #
    # _experiments = list(
    #     filter(
    #         lambda experiment: experiment.config_valid == config_valid,
    #         _experiments,
    #     )
    # )
    #
    # _experiments = list(
    #     filter(
    #         lambda experiment: experiment.outputs_valid == outputs_valid,
    #         _experiments,
    #     )
    # )
    #
    # _experiments = list(
    #     filter(
    #         lambda experiment: experiment.results_valid == results_valid,
    #         _experiments,
    #     )
    # )

    if dataset is not None:
        _experiments = list(
            filter(
                lambda experiment: experiment.dataset == dataset,
                _experiments,
            )
        )

    if dropout is not None:
        _experiments = list(
            filter(
                lambda experiment: experiment.dropout == dropout,
                _experiments,
            )
        )
    if rew is not None:
        _experiments = list(
            filter(
                lambda experiment: experiment.reward == rew,
                _experiments,
            )
        )
    if run_nr is not None:
        _experiments = list(
            filter(
                lambda experiment: experiment.run == run_nr,
                _experiments,
            )
        )

    if model is not None:
        _experiments = list(
            filter(
                lambda experiment: experiment.model == model,
                _experiments,
            )
        )

    if backbone is not None:
        _experiments = list(
            filter(
                lambda experiment: experiment.backbone == backbone,
                _experiments,
            )
        )

    if exclude_model is not None:
        _experiments = list(
            filter(
                lambda experiment: experiment.model != exclude_model,
                _experiments,
            )
        )

    # if n is not None:
    #     _experiments = _experiments[:n]

    # if precision_study:
    #     _experiments = list(
    #         filter(
    #             lambda experiment: "precision_study" in str(experiment.group_dir)
    #             and "64" not in str(experiment.group_dir),
    #             _experiments,
    #         )
    #     )
    # else:
    #     _experiments = list(
    #         filter(
    #             lambda experiment: "precision_study" not in str(experiment.group_dir),
    #             _experiments,
    #         )
    #     )

    if name is not None:
        _experiments = list(
            filter(
                lambda experiment: str(experiment.to_path()) == name,
                _experiments,
            )
        )

    # if not ignore_running:
    # jobs = get_jobs()

    # _experiments = list(
    #     filter(lambda e: not is_experiment_running(e, jobs), _experiments)
    # )

    print("Launching:")
    for exp in map(
        lambda exp: str(exp.to_path()),
        _experiments,
    ):
        rich.print(exp)

    if not dry_run:
        # if local:
        if cluster:
            submit(_experiments, mode)
        # run(_experiments, mode)
        else:
            asyncio.run(run(_experiments, mode))
        # else:
        #     submit(_experiments, mode)


def add_arguments(parser: argparse.ArgumentParser):
    sub_parsers = parser.add_subparsers()

    # parser.add_argument("--validation-file", default=None, type=Path)
    # parser.add_argument("--config-exists", action="store_true")
    # parser.add_argument("--model-exists", action="store_true")
    # parser.add_argument("--config-valid", action="store_true")
    # parser.add_argument("--outputs-valid", action="store_true")
    # parser.add_argument("--results-valid", action="store_true")

    # parser.add_argument("--study", default=None, type=str)
    parser.add_argument("--dataset", default=None, type=str)
    parser.add_argument("--dropout", default=None, type=int, choices=(0, 1))
    parser.add_argument(
        "--model", default=None, type=str, choices=("vit", "dg", "devries", "confidnet")
    )
    parser.add_argument("--backbone", default=None, type=str, choices=("vit",))
    parser.add_argument(
        "--exclude-model",
        default=None,
        type=str,
        choices=("vit", "dg", "devries", "confidnet"),
    )
    # parser.add_argument("--precision-study", action="store_true")

    # parser.add_argument("-n", "--limit", default=None, type=int)
    parser.add_argument("--run", default=None, type=int)
    parser.add_argument("--reward", default=None, type=float)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--mode",
        default="train_test",
        choices=("test", "train", "train_test", "analysis"),
    )
    parser.add_argument("--cluster", action="store_true")
    # parser.add_argument("--ignore-running", action="store_true")
    # parser.add_argument("--jobs-list", default=None, type=Path)

    parser.add_argument("--name", default=None, type=str)

    return parser


def main(args):
    # jobs_list: list[str] | None = None
    #
    # if args.jobs_list is not None:
    #     with open(args.jobs_list, "rt") as f:
    #         jobs_list = f.read().split("\n")

    launch(
        # validation_file=args.validation_file,
        # study=args.study,
        dataset=args.dataset,
        dropout=args.dropout,
        model=args.model,
        backbone=args.backbone,
        exclude_model=args.exclude_model,
        # config_exists=args.config_exists,
        # model_exists=args.model_exists,
        # config_valid=args.config_valid,
        # outputs_valid=args.outputs_valid,
        # results_valid=args.results_valid,
        mode=args.mode,
        dry_run=args.dry_run,
        run_nr=args.run,
        rew=args.reward,
        # precision_study=args.precision_study,
        cluster=args.cluster,
        # ignore_running=args.ignore_running,
        # jobs_list=jobs_list,
        name=args.name,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args()

    main(args)
