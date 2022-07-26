from pathlib import Path
import subprocess

VERSION = "0.0.1"

def _git_rev():
    return subprocess.check_output(['git', 'describe', '--always'], cwd=Path(__file__).parent).rstrip().decode('utf8')

def version():
    return f"{VERSION}+{_git_rev()}"
