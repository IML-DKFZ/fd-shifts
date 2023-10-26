import argparse

from fd_shifts import reporting
from fd_shifts.experiments import get_all_experiments, launcher
from fd_shifts.loaders import prepare


def _list_experiments(args) -> None:
    _experiments = launcher.filter_experiments(
        dataset=args.dataset,
        dropout=args.dropout,
        model=args.model,
        backbone=args.backbone,
        exclude_model=args.exclude_model,
        run_nr=args.run,
        rew=args.reward,
        name=args.name,
    )

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
    launcher.add_filter_arguments(list_parser)
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


if __name__ == "__main__":
    main()
