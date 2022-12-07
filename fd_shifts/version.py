from importlib.metadata import PackageNotFoundError, version


def get_version() -> str:
    """Error handled wrapper around importlib's version

    Returns:
        the installed version of fd_shifts
    """
    try:
        return version("fd_shifts")
    except PackageNotFoundError:
        # package is not installed
        import setuptools_scm

        return setuptools_scm.get_version(root="..", relative_to=__file__)
