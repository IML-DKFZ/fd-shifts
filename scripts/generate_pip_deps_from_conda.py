#!/usr/bin/env python3
"""
Adapted from https://github.com/pandas-dev/pandas

Convert the conda environment.yaml to the pip requirements.txt,
or check that they have the same packages (for the CI)

Usage:

    Generate `requirements-dev.txt`
    $ python scripts/generate_pip_deps_from_conda.py

    Compare and fail (exit status != 0) if `requirements-dev.txt` has not been
    generated with this script:
    $ python scripts/generate_pip_deps_from_conda.py --compare
"""
import argparse
import pathlib
import re
import sys

import toml
import yaml

EXCLUDE = {"python", "c-compiler", "cxx-compiler", "pip", "yaml"}
RENAME = {"pytorch": "torch"}


def conda_package_to_pip(package: str):
    """
    Convert a conda package to its pip equivalent.

    In most cases they are the same, those are the exceptions:
    - Packages that should be excluded (in `EXCLUDE`)
    - Packages that should be renamed (in `RENAME`)
    - A package requiring a specific version, in conda is defined with a single
      equal (e.g. ``pandas=1.0``) and in pip with two (e.g. ``pandas==1.0``)
    """
    package = re.sub("(?<=[^<>])=", "==", package).strip()

    for compare in ("<=", ">=", "=="):
        if compare in package:
            pkg, version = package.split(compare)
            if pkg in EXCLUDE:
                return

            if pkg in RENAME:
                return "".join((RENAME[pkg], compare, version))

    if package in EXCLUDE:
        return

    if package in RENAME:
        return RENAME[package]

    return package


def generate_pip_from_conda(
    conda_path: pathlib.Path, pip_path: pathlib.Path, compare: bool = False
) -> bool:
    """
    Generate the pip dependencies file from the conda file, or compare that
    they are synchronized (``compare=True``).

    Parameters
    ----------
    conda_path : pathlib.Path
        Path to the conda file with dependencies (e.g. `environment.yml`).
    pip_path : pathlib.Path
        Path to the pip file with dependencies (e.g. `requirements-dev.txt`).
    compare : bool, default False
        Whether to generate the pip file (``False``) or to compare if the
        pip file has been generated with this script and the last version
        of the conda file (``True``).

    Returns
    -------
    bool
        True if the comparison fails, False otherwise
    """
    with conda_path.open() as file:
        deps = yaml.safe_load(file)["dependencies"]

    pip_deps = []
    for dep in deps:
        if isinstance(dep, str):
            conda_dep = conda_package_to_pip(dep)
            if conda_dep:
                pip_deps.append(conda_dep)
        elif isinstance(dep, dict) and len(dep) == 1 and "pip" in dep:
            pip_deps.extend(dep["pip"])
        else:
            raise ValueError(f"Unexpected dependency {dep}")

    header = (
        f"# This file is auto-generated from {conda_path.name}, do not modify.\n"
        "# See that file for comments about the need/usage of each dependency.\n\n"
    )
    pip_content = header + "\n".join(pip_deps) + "\n"

    if compare:
        with pip_path.open() as file:
            return pip_content != file.read()

    with pip_path.open("w") as file:
        file.write(pip_content)
    return False


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="convert (or compare) conda file to pip"
    )
    argparser.add_argument(
        "--compare",
        action="store_true",
        help="compare whether the two files are equivalent",
    )
    args = argparser.parse_args()

    conda_fname = "environment.yaml"
    pip_fname = "requirements.txt"
    repo_path = pathlib.Path(__file__).parent.parent.absolute()
    res = generate_pip_from_conda(
        pathlib.Path(repo_path, conda_fname),
        pathlib.Path(repo_path, pip_fname),
        compare=args.compare,
    )
    if res:
        msg = (
            f"`{pip_fname}` has to be generated with `{__file__}` after "
            f"`{conda_fname}` is modified.\n"
        )
        sys.stderr.write(msg)
    sys.exit(res)
