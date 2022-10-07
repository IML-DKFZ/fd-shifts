import subprocess
from pathlib import Path

BASH_RSYNC_COMMAND = r"""
rsync --relative /home/t974t/NetworkDrives/E130-Personal/Bungert/./{dir}/ odcf-worker01:/dkfz/cluster/gpu/checkpoints/OE0612/t974t/ --exclude='*.npz*' -azzhuv --info=progress2
rsync --relative /media/experiments/./{dir}/ odcf-worker01:/dkfz/cluster/gpu/checkpoints/OE0612/t974t/ --exclude='*.npz*' -azzhuv --info=progress2
"""


def sync_to_remote(directory: Path):
    subprocess.run(BASH_RSYNC_COMMAND.format(dir=directory).strip(), shell=True, check=True)
