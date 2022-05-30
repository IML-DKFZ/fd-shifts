import socket
import subprocess
from pathlib import Path
import time

from omegaconf.dictconfig import DictConfig

IS_CLUSTER = socket.gethostname().startswith("odcf")
CMD_PATH = Path(__file__).parent.parent / "fd_shifts/exec.py"

if IS_CLUSTER:
    base_path = Path("/dkfz/cluster/gpu/checkpoints/fd-shifts")
    base_command = " \\\n".join(
        [
            "bsub",
            "-gpu num=1:j_exclusive=yes:gmem=10.7G",
            "-L /bin/bash -q gpu",
            "-u 'till.bungert@dkfz-heidelberg.de' -B -N",
            "-g /t974t/test",
            '-J "{exp_name}_test"',
            (
                "'source ~/.env && conda activate $CONDA_ENV/fd-shifts && "
                "python -W ignore {cmd} {args}'"
            ),
        ]
    )
else:
    base_path = Path("~/Experiments/fd-shifts").expanduser()
    base_command = (
        "echo {exp_name} && "
        "bash -li -c "
        "'source ~/.bashrc && "
        "micromamba activate fd-shifts "
        "&& EXPERIMENT_ROOT_DIR=/home/t974t/Experiments/fd-shifts "
        "DATASET_ROOT_DIR=/home/t974t/Data "
        "python -W ignore {cmd} {args}'"
    )


for path in base_path.glob("**/raw_output.npz"):
    exp_name = path.parent.parent.parts[-1]

    args = [f"--config-path={path.parent.parent / 'hydra/'}", "exp.mode=test"]
    args.append('++exp.output_paths.test.raw_output="\\${test.dir}/raw_logits.npz"')
    args.append(
        '++exp.output_paths.test.raw_output_dist="\\${test.dir}/raw_logits_dist.npz"'
    )

    launch_command = base_command.format(
        exp_name=exp_name, cmd=CMD_PATH, args=" ".join(args)
    )
    print("Launch command: ", launch_command, end="\n\n")
    subprocess.run(launch_command, shell=True, check=True)
    time.sleep(1)
    break
