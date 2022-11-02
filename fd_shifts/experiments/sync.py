import subprocess
from pathlib import Path

from rich import print
from rich.syntax import Syntax

BASH_RSYNC_COMMAND = r"""
# rsync --relative /home/t974t/NetworkDrives/E130-Personal/Bungert/./{dir}/ odcf-worker01:/dkfz/cluster/gpu/checkpoints/OE0612/t974t/ --exclude='*.npz*' -azzhuv --info=progress2
rsync --relative /media/experiments/./{dir}/ odcf-worker01:/dkfz/cluster/gpu/checkpoints/OE0612/t974t/ --include='*/' --include='last.ckpt' --exclude='*' -azzhuv --info=progress2
"""


def sync_to_dir_remote(directory: Path, dry_run: bool = False):
    print(
        Syntax(
            BASH_RSYNC_COMMAND.format(dir=directory).strip(),
            "bash",
            word_wrap=True,
            background_color="default",
        )
    )

    if not dry_run:
        subprocess.run(
            BASH_RSYNC_COMMAND.format(dir=directory).strip(), shell=True, check=True
        )
