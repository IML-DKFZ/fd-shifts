import argparse

from fd_shifts import experiments
# import fd_shifts.exec as fd_shifts_exec
from fd_shifts.experiments import launcher
from fd_shifts import reporting


def list_experiments(_):
    _experiments = experiments.get_all_experiments()
    for exp in _experiments:
        print(exp.to_path())


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title="commands")
    parser.set_defaults(command=lambda _: parser.print_help())

    list_parser = subparsers.add_parser("list")
    list_parser.set_defaults(command=list_experiments)

    launch_parser = subparsers.add_parser("launch")
    launcher.add_arguments(launch_parser)
    launch_parser.set_defaults(command=launcher.main)

    reporting_parser = subparsers.add_parser("reporting")
    reporting_parser.set_defaults(command=lambda _: reporting.main("./reporting"))

    args = parser.parse_args()
    args.command(args)
