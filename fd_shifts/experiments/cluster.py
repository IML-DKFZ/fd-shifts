import re
import subprocess
from typing import Any

from rich import print
from rich.syntax import Syntax

from fd_shifts import experiments

# -R "select[hname!='e230-dgx2-2']" \

BASH_BSUB_COMMAND = r"""
bsub -gpu num=1:j_exclusive=yes:gmem={gmem}\
    -L /bin/bash \
    -q gpu \
    -u 'till.bungert@dkfz-heidelberg.de' \
    -B {nodes} \
    -g /t974t/train \
    -J "{name}" \
    bash -li -c 'set -o pipefail; echo $LSB_JOBID && source .envrc && {command} |& tee -a "/home/t974t/logs/$LSB_JOBID.log"'
"""

BASH_BASE_COMMAND = r"""
_fd_shifts_exec {overrides} exp.mode={mode}
"""


def get_nodes(mode: str):
    match mode:
        case "train" | "train_test":
            return "-sp 36"
        case _:
            return ""


def get_gmem(mode: str, model: str):
    match mode:
        case "train" | "train_test":
            match model:
                case "vit":
                    return "23G"
                case _:
                    return "23G"
        case _:
            match model:
                case "vit":
                    return "23G"
                case _:
                    return "23G"


def update_overrides(
    overrides: dict[str, Any], iid_only: bool = False, mode: str = "train_test"
) -> dict[str, Any]:
    if mode in ["train", "train_test"] and overrides.get("trainer.batch_size", -1) > 32:
        accum = overrides["trainer.batch_size"] // 32
        overrides["trainer.batch_size"] = 32
        overrides["trainer.accumulate_grad_batches"] = accum

    if mode in ["test"]:
        overrides["trainer.batch_size"] = 256

    if iid_only:
        overrides["eval.query_studies.noise_study"] = []
        overrides["eval.query_studies.in_class_study"] = []
        overrides["eval.query_studies.new_class_study"] = []

    return overrides


def submit(
    _experiments: list[experiments.Experiment], mode: str, dry_run: bool, iid_only: bool
):
    try:
        from pssh.clients import SSHClient
        from pssh.exceptions import Timeout
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "You need to run pip install parallel-ssh to submit to the cluster"
        ) from exc

    if len(_experiments) == 0:
        print("Nothing to run")
        return

    if not dry_run:
        client = SSHClient("odcf-worker02.inet.dkfz-heidelberg.de")

    for experiment in _experiments:
        try:
            # if path := experiment.overrides().get(
            #     "trainer.callbacks.training_stages.pretrained_backbone_path"
            # ):
            #     sync_to_dir_remote(
            #         path.replace("${EXPERIMENT_ROOT_DIR%/}/", "fd-shifts/"),
            #         dry_run=dry_run,
            #     )

            overrides = update_overrides(
                experiment.overrides(), iid_only=iid_only, mode=mode
            )
            cmd = BASH_BASE_COMMAND.format(
                overrides=" ".join(f"{k}={v}" for k, v in overrides.items()),
                mode=mode,
            ).strip()

            print(
                Syntax(
                    re.sub(r"([^,]) ", "\\1 \\\n\t", cmd),
                    "bash",
                    word_wrap=True,
                    background_color="default",
                )
            )

            cmd = BASH_BSUB_COMMAND.format(
                name=experiment.to_path().relative_to("fd-shifts"),
                command=cmd,
                nodes=get_nodes(mode),
                gmem=get_gmem(mode, experiment.model),
            ).strip()

            print(
                Syntax(
                    cmd,
                    "bash",
                    word_wrap=True,
                    background_color="default",
                )
            )

            if dry_run:
                continue

            with client.open_shell(read_timeout=1) as shell:
                shell.run("cd ~/Projects/failure-detection-benchmark")
                shell.run("source .envrc")
                shell.run(cmd)

                try:
                    for line in shell.stdout:
                        print(line)
                except Timeout:
                    pass

                try:
                    for line in shell.stderr:
                        print(line)
                except Timeout:
                    pass

            for line in shell.stdout:
                print(line)

            for line in shell.stderr:
                print(line)

        except subprocess.CalledProcessError:
            continue
