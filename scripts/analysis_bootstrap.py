import argparse
import asyncio
import multiprocessing
import subprocess
from datetime import datetime
from pathlib import Path

import rich
from rich.syntax import Syntax
from tqdm import tqdm

from fd_shifts.experiments.launcher import (
    add_filter_arguments,
    filter_experiments,
    get_filter,
)

BASH_LOCAL_COMMAND = r"""
bash -c 'set -o pipefail; {command} |& tee -a "./logs_bootstrap/{log_file_name}.log"'
"""

BASH_BASE_COMMAND = r"""
fd-shifts analysis_bootstrap \
    --experiment={experiment} \
    --n_bs={n_bs} \
    --exclude_noise_study={exclude_noise_study} \
    --no_iid={no_iid} \
    --iid_only={iid_only} {overrides}
"""


def run_command(command):
    subprocess.run(command, shell=True)


async def run_experiments(
    _experiments: list[str],
    dry_run: bool,
    iid_only: bool = False,
    no_iid: bool = False,
    exclude_noise_study: bool = False,
    n_bs: int = 500,
    num_workers: int = 12,
):
    if len(_experiments) == 0:
        print("Nothing to run")
        return

    Path("./logs").mkdir(exist_ok=True)

    queue = []

    for experiment in _experiments:
        log_file_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{experiment.replace('/', '_').replace('.','_')}"

        overrides = {}

        cmd = BASH_BASE_COMMAND.format(
            experiment=experiment,
            n_bs=n_bs,
            iid_only=iid_only,
            no_iid=no_iid,
            exclude_noise_study=exclude_noise_study,
            overrides=" ".join(f"--config.{k}={v}" for k, v in overrides.items()),
        ).strip()

        cmd = BASH_LOCAL_COMMAND.format(
            command=cmd, log_file_name=log_file_name
        ).strip()
        rich.print(Syntax(cmd, "bash", word_wrap=True, background_color="default"))
        if not dry_run:
            queue.append(cmd)

    if queue == []:
        return

    # Create a tqdm progress bar
    with tqdm(total=len(queue), desc="Experiments") as pbar:
        # Create a pool of worker processes
        pool = multiprocessing.Pool(processes=num_workers)
        # Map the list of commands to the worker pool
        for _ in pool.imap_unordered(run_command, queue):
            pbar.update()
        # Close the pool to prevent any more tasks from being submitted
        pool.close()
        # Wait for all processes to finish
        pool.join()


def launch(
    dataset: str | None,
    dropout: int | None,
    model: str | None,
    backbone: str | None,
    exclude_model: str | None,
    exclude_backbone: str | None,
    exclude_group: str | None,
    dry_run: bool,
    run: int | None,
    reward: float | None,
    cluster: bool,
    experiment: str | None,
    iid_only: bool,
    no_iid: bool,
    exclude_noise_study: bool,
    n_bs: int,
    num_workers: int,
    custom_filter: str | None,
):
    _experiments = list(
        filter_experiments(
            dataset,
            dropout,
            model,
            backbone,
            exclude_model,
            exclude_backbone,
            exclude_group,
            run,
            reward,
            experiment,
        )
    )

    if custom_filter is not None:
        print(f"Applying custom filter {custom_filter}...")
        _experiments = get_filter(custom_filter)(_experiments)

    _experiments = list(_experiments)

    print(f"Launching {len(_experiments)} experiments:")
    for exp in _experiments:
        rich.print(exp)

    if cluster:
        raise NotImplementedError()
    else:
        asyncio.run(
            run_experiments(
                _experiments,
                dry_run,
                iid_only,
                no_iid,
                exclude_noise_study,
                n_bs,
                num_workers,
            )
        )


def add_arguments(parser: argparse.ArgumentParser):
    add_filter_arguments(parser)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--iid-only", action="store_true")
    parser.add_argument("--no_iid", action="store_true")
    parser.add_argument("--exclude-noise-study", action="store_true")
    parser.add_argument("--n-bs", default=500, type=int)
    parser.add_argument("--num-workers", default=2, type=int)
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args()

    launch(
        dataset=args.dataset,
        dropout=args.dropout,
        model=args.model,
        backbone=args.backbone,
        exclude_model=args.exclude_model,
        exclude_backbone=args.exclude_backbone,
        exclude_group=args.exclude_group,
        dry_run=args.dry_run,
        run=args.run,
        reward=args.reward,
        cluster=args.cluster,
        experiment=args.experiment,
        iid_only=args.iid_only,
        no_iid=args.no_iid,
        exclude_noise_study=args.exclude_noise_study,
        n_bs=args.n_bs,
        num_workers=args.num_workers,
        custom_filter=args.custom_filter,
    )
