import subprocess
from pathlib import Path

# lang: bash
RSYNC_COMMAND = r"""
rsync --relative /home/t974t/Experiments/./{dir}/ odcf-worker01:/dkfz/cluster/gpu/checkpoints/OE0612/t974t/ -azzhuv --info=progress2
"""


def sync_to_remote(directory: Path):
    subprocess.run(RSYNC_COMMAND.format(dir=directory).strip(), shell=True, check=True)
