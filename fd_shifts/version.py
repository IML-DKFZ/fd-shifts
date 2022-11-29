from importlib.metadata import PackageNotFoundError, version


def get_version() -> str:
    try:
        return version("fd_shifts")
    except PackageNotFoundError:
        # package is not installed
        import setuptools_scm

        return setuptools_scm.get_version(root="..", relative_to=__file__)
