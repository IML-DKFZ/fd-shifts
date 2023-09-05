import argparse

from fd_shifts import experiments, reporting
from fd_shifts.experiments import launcher
from fd_shifts.loaders import prepare


def _list_experiments(_) -> None:
    _experiments = experiments.get_all_experiments()
    for exp in _experiments:
        print(exp.to_path())


def main() -> None:
    """Entry point for the command line interface

    This gets installed as a script named `fd_shifts` by pip.
    """
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title="commands")
    parser.set_defaults(command=lambda _: parser.print_help())

    list_parser = subparsers.add_parser("list")
    list_parser.set_defaults(command=_list_experiments)

    launch_parser = subparsers.add_parser("launch")
    launcher.add_arguments(launch_parser)
    launch_parser.set_defaults(command=launcher.main)

    reporting_parser = subparsers.add_parser("reporting")
    reporting_parser.set_defaults(command=lambda _: reporting.main("./results"))

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser = prepare.add_arguments(prepare_parser)
    prepare_parser.set_defaults(command=lambda _: prepare.main)

    args = parser.parse_args()
    args.command(args)
