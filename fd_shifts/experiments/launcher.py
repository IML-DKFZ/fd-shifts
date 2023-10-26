import argparse
import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import rich
from rich.syntax import Syntax

from fd_shifts import experiments, logger
from fd_shifts.experiments.cluster import submit

BASH_LOCAL_COMMAND = r"""
bash -c 'set -o pipefail; {command} |& tee -a "./logs/{log_file_name}.log"'
"""

BASH_BASE_COMMAND = r"""
_fd_shifts_exec {overrides} exp.mode={mode}
"""


async def worker(name, queue: asyncio.Queue[str]):
    while True:
        # Get a "work item" out of the queue.
        cmd = await queue.get()
        logger.info(f"{name} running {cmd}")
        proc = await asyncio.create_subprocess_shell(
            cmd,
        )

        # Wait for the subprocess exit.
        await proc.wait()

        if proc.returncode != 0:
            logger.error(f"{name} running {cmd} finished abnormally")
        else:
            logger.info(f"{name} running {cmd} finished")

        # Notify the queue that the "work item" has been processed.
        queue.task_done()


def update_overrides(
    overrides: dict[str, Any], iid_only: bool = False, mode: str = "train_test"
) -> dict[str, Any]:
    if mode in ["train", "train_test"] and overrides.get("trainer.batch_size", -1) > 32:
        accum = overrides["trainer.batch_size"] // 32
        overrides["trainer.batch_size"] = 32
        overrides["trainer.accumulate_grad_batches"] = accum

    if mode in ["test"]:
        overrides["trainer.batch_size"] = 256

    if iid_only:
        overrides["eval.query_studies.noise_study"] = []
        overrides["eval.query_studies.in_class_study"] = []
        overrides["eval.query_studies.new_class_study"] = []

    return overrides


async def run(
    _experiments: list[experiments.Experiment],
    mode: str,
    dry_run: bool,
    max_batch_size: int = 32,
    iid_only: bool = False,
):
    if len(_experiments) == 0:
        print("Nothing to run")
        return

    Path("./logs").mkdir(exist_ok=True)

    # Create a queue that we will use to store our "workload".
    queue: asyncio.Queue[str] = asyncio.Queue()

    for experiment in _experiments:
        log_file_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{str(experiment.to_path()).replace('/', '_').replace('.','_')}"

        overrides = update_overrides(experiment.overrides(), iid_only, mode)

        cmd = BASH_BASE_COMMAND.format(
            overrides=" ".join(f"{k}={v}" for k, v in overrides.items()),
            mode=mode,
        ).strip()

        print(
            Syntax(
                re.sub(r"([^,]) ", "\\1 \\\n\t", cmd),
                "bash",
                word_wrap=True,
                background_color="default",
            )
        )

        cmd = BASH_LOCAL_COMMAND.format(
            command=cmd, log_file_name=log_file_name
        ).strip()
        print(Syntax(cmd, "bash", word_wrap=True, background_color="default"))
        if not dry_run:
            queue.put_nowait(cmd)

        break

    if queue.empty():
        return

    tasks = []
    for i in range(1):
        task = asyncio.create_task(worker(f"worker-{i}", queue))
        tasks.append(task)

    # Wait until the queue is fully processed.
    await queue.join()

    # Cancel our worker tasks.
    for task in tasks:
        task.cancel()
    # Wait until all worker tasks are cancelled.
    await asyncio.gather(*tasks, return_exceptions=True)


def filter_experiments(
    dataset: str | None,
    dropout: int | None,
    model: str | None,
    backbone: str | None,
    exclude_model: str | None,
    run_nr: int | None,
    rew: float | None,
    name: str | None,
) -> Iterator[experiments.Experiment]:
    _experiments = experiments.get_all_experiments()

    _experiments = filter(
        lambda e: "precision_study" not in str(e.to_path()), _experiments
    )

    if dataset is not None:
        _experiments = filter(
            lambda experiment: experiment.dataset == dataset,
            _experiments,
        )

    if dropout is not None:
        _experiments = filter(
            lambda experiment: experiment.dropout == dropout,
            _experiments,
        )
    if rew is not None:
        _experiments = filter(
            lambda experiment: experiment.reward == rew,
            _experiments,
        )
    if run_nr is not None:
        _experiments = filter(
            lambda experiment: experiment.run == run_nr,
            _experiments,
        )

    if model is not None:
        _experiments = filter(
            lambda experiment: experiment.model == model,
            _experiments,
        )

    if backbone is not None:
        _experiments = filter(
            lambda experiment: experiment.backbone == backbone,
            _experiments,
        )

    if exclude_model is not None:
        _experiments = filter(
            lambda experiment: experiment.model != exclude_model,
            _experiments,
        )

    if name is not None:
        _experiments = filter(
            lambda experiment: str(experiment.to_path()) == name,
            _experiments,
        )

    return _experiments


def launch(
    dataset: str | None,
    dropout: int | None,
    model: str | None,
    backbone: str | None,
    exclude_model: str | None,
    mode: str,
    dry_run: bool,
    run_nr: int | None,
    rew: float | None,
    cluster: bool,
    name: str | None,
    max_batch_size: int,
    iid_only: bool,
):
    _experiments = filter_experiments(
        dataset,
        dropout,
        model,
        backbone,
        exclude_model,
        run_nr,
        rew,
        name,
    )
    print("Launching:")
    for exp in map(
        lambda exp: str(exp.to_path()),
        _experiments,
    ):
        rich.print(exp)

    if cluster:
        submit(_experiments, mode, dry_run, iid_only)
    else:
        asyncio.run(run(_experiments, mode, dry_run, max_batch_size, iid_only))


def add_filter_arguments(parser: argparse.ArgumentParser):
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

    parser.add_argument("--run", default=None, type=int)
    parser.add_argument("--reward", default=None, type=float)
    parser.add_argument("--name", default=None, type=str)

    return parser


def add_arguments(parser: argparse.ArgumentParser):
    add_filter_arguments(parser)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--mode",
        default="train_test",
        choices=("test", "train", "train_test", "analysis"),
    )
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--iid-only", action="store_true")

    parser.add_argument("--max-batch-size", default=32, type=int)

    return parser


def main(args):
    #

    launch(
        dataset=args.dataset,
        dropout=args.dropout,
        model=args.model,
        backbone=args.backbone,
        exclude_model=args.exclude_model,
        mode=args.mode,
        dry_run=args.dry_run,
        run_nr=args.run,
        rew=args.reward,
        cluster=args.cluster,
        name=args.name,
        max_batch_size=args.max_batch_size,
        iid_only=args.iid_only,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args()

    main(args)
