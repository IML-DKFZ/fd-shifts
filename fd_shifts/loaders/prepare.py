import argparse
import os
from pathlib import Path

import imageio.core.util

from fd_shifts.loaders.preparation import (
    prepare_dermoscopy,
    prepare_lidc,
    prepare_rxrx1,
    prepare_xray,
)


def ignore_warnings(*args, **kwargs):
    pass


imageio.core.util._precision_warn = ignore_warnings


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--dataset",
        default="all",
        choices=("all", "xray", "microscopy", "dermoscopy", "lung_ct"),
    )

    return parser


def main(args: argparse.Namespace):
    data_dir = Path(os.getenv("DATASET_ROOT_DIR", "./data"))
    if args.dataset == "all" or args.dataset == "microscopy":
        prepare_rxrx1(data_dir)
    if args.dataset == "all" or args.dataset == "xray":
        prepare_xray(data_dir)
    if args.dataset == "all" or args.dataset == "dermoscopy":
        prepare_dermoscopy(data_dir)
    if args.dataset == "all" or args.dataset == "lung_ct":
        prepare_lidc(data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args()

    main(args)
