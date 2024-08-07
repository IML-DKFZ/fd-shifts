import argparse
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any

import rich
from rich.syntax import Syntax
from tqdm import tqdm

from fd_shifts import logger
from fd_shifts.experiments.configs import get_experiment_config, list_experiment_configs

BASH_LOCAL_COMMAND = r"""
bash -c 'set -o pipefail; {command} |& tee -a "./logs/{log_file_name}.log"'
"""

BASH_BASE_COMMAND = r"""
fd-shifts {mode} --experiment={experiment} {overrides}
"""


async def worker(name, queue: asyncio.Queue[str], progress_bar=None):
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

        if progress_bar is not None:
            progress_bar.update(1)

        # Notify the queue that the "work item" has been processed.
        queue.task_done()


async def run_experiments(
    _experiments: list[str],
    mode: str,
    dry_run: bool,
    override: dict | None,
):
    if len(_experiments) == 0:
        print("Nothing to run")
        return

    Path("./logs").mkdir(exist_ok=True)
    override = override if override is not None else {}

    # Create a queue that we will use to store our "workload".
    queue: asyncio.Queue[str] = asyncio.Queue()

    for experiment in _experiments:
        log_file_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{experiment.replace('/', '_').replace('.','_')}"
        cmd = BASH_BASE_COMMAND.format(
            experiment=experiment,
            overrides=" ".join([f"--{k}={v}" for k, v in override.items()]),
            mode=mode,
        ).strip()

        cmd = BASH_LOCAL_COMMAND.format(
            command=cmd, log_file_name=log_file_name
        ).strip()
        rich.print(Syntax(cmd, "bash", word_wrap=True, background_color="default"))
        if not dry_run:
            queue.put_nowait(cmd)

    if queue.empty():
        return

    progress_bar = tqdm(total=queue.qsize(), desc="Experiments")

    tasks = []
    for i in range(1):
        task = asyncio.create_task(worker(f"worker-{i}", queue, progress_bar))
        tasks.append(task)

    # Wait until the queue is fully processed.
    await queue.join()

    # Cancel our worker tasks.
    for task in tasks:
        task.cancel()
    # Wait until all worker tasks are cancelled.
    await asyncio.gather(*tasks, return_exceptions=True)


def launch(args):
    _experiments = filter_experiments(
        dataset=args.dataset,
        dropout=args.dropout,
        model=args.model,
        backbone=args.backbone,
        exclude_model=args.exclude_model,
        exclude_backbone=args.exclude_backbone,
        exclude_group=args.exclude_group,
        run_nr=args.run,
        rew=args.reward,
        experiment=args.experiment,
        custom_filter=args.custom_filter,
    )
    logger.info(f"Launching {len(_experiments)} experiments:")
    for exp in _experiments:
        logger.info(exp)

    if args.cluster:
        raise NotImplementedError()
    else:
        asyncio.run(
            run_experiments(_experiments, args.mode, args.dry_run, args.override)
        )


def filter_experiments(
    dataset: str | None,
    dropout: int | None,
    model: str | None,
    backbone: str | None,
    exclude_model: str | None,
    exclude_backbone: str | None,
    exclude_group: str | None,
    run_nr: int | None,
    rew: float | None,
    experiment: str | None,
    custom_filter: str | None,
) -> filter:
    _experiments = list_experiment_configs()

    if exclude_group is not None:
        _experiments = filter(
            lambda e: get_experiment_config(e).exp.group_name != exclude_group,
            _experiments,
        )

    if dataset is not None:
        _experiments = filter(
            lambda e: get_experiment_config(e).data.dataset == dataset,
            _experiments,
        )

    if dropout is not None:
        _experiments = filter(
            lambda e: get_experiment_config(e).model.dropout_rate == dropout,
            _experiments,
        )
    if rew is not None:
        _experiments = filter(
            lambda e: get_experiment_config(e).model.dg_reward == rew,
            _experiments,
        )
    if run_nr is not None:
        _experiments = filter(
            lambda e: f"_run{run_nr}_" in e,
            _experiments,
        )

    if model is not None:
        _experiments = filter(
            lambda e: get_experiment_config(e).model.name == model + "_model",
            _experiments,
        )

    if backbone is not None:
        _experiments = filter(
            lambda e: get_experiment_config(e).model.network.name == backbone,
            _experiments,
        )

    if exclude_model is not None:
        _experiments = filter(
            lambda e: get_experiment_config(e).model.name != exclude_model + "_model",
            _experiments,
        )

    if exclude_backbone is not None:
        _experiments = filter(
            lambda e: get_experiment_config(e).model.network.name != exclude_backbone,
            _experiments,
        )

    if experiment is not None:
        _experiments = filter(lambda e: e == experiment, _experiments)

    if custom_filter is not None:
        logger.info(f"Applying custom filter {custom_filter}...")
        _experiments = get_filter(custom_filter)(_experiments)

    return _experiments


_FILTERS = {}


def register_filter(name):
    def _inner_wrapper(func):
        _FILTERS[name] = func
        return func

    return _inner_wrapper


def get_filter(name):
    try:
        return _FILTERS[name]
    except KeyError as err:
        raise ValueError(
            f"Filter name '{name}' not valid. Available filters: {', '.join(_FILTERS)}"
        ) from err


@register_filter("iclr2023")
def filter_iclr2023(experiments):
    from fd_shifts.experiments.publications import ICLR2023

    def is_valid(exp):
        return exp in ICLR2023

    return filter(is_valid, experiments)


@register_filter("augrc2024")
def filter_augrc2024(experiments):
    from fd_shifts.experiments.publications import AUGRC2024

    def is_valid(exp):
        return exp in AUGRC2024

    return filter(is_valid, experiments)


def add_filter_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--dataset", default=None, type=str)
    parser.add_argument("--dropout", default=None, type=int, help="0 or 1")
    parser.add_argument(
        "--model", default=None, type=str, choices=("vit", "dg", "devries", "confidnet")
    )
    parser.add_argument("--backbone", default=None, type=str, choices=("vit",))
    parser.add_argument("--exclude-backbone", default=None, type=str)
    parser.add_argument("--exclude-group", default=None, type=str)
    parser.add_argument(
        "--exclude-model",
        default=None,
        type=str,
        choices=("vit", "dg", "devries", "confidnet"),
    )
    parser.add_argument("--run", default=None, type=int)
    parser.add_argument("--reward", default=None, type=float)
    parser.add_argument("--experiment", default=None, type=str)
    parser.add_argument("--custom-filter", default=None, type=str, choices=_FILTERS)
    return parser


def add_launch_arguments(parser: argparse.ArgumentParser):
    add_filter_arguments(parser)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mode", required=True, choices=("train", "test", "analysis"))
    parser.add_argument("--cluster", action="store_true")
    # https://jsonargparse.readthedocs.io/en/stable/#dict-items
    parser.add_argument(
        "--override",
        type=dict,
        default=None,
        help=(
            "Additional configurations passed to each `fd-shifts {mode}` call can be "
            "specified via `--override.key=value` (e.g. "
            "`override.config.trainer.batch_size=64`)"
        ),
    )
    return parser
