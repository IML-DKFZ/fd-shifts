import subprocess
from pathlib import Path

VERSION = "0.0.1"


def _git_rev() -> str:
    return (
        subprocess.check_output(
            ["git", "describe", "--always"], cwd=Path(__file__).parent
        )
        .rstrip()
        .decode("utf8")
    )


def version() -> str:
    """Create version string including git commit hash

    Returns:
        Version string
    """
    return f"{VERSION}+{_git_rev()}"
