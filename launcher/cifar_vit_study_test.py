import os
import subprocess
import time
from itertools import product

current_dir = os.path.dirname(os.path.realpath(__file__))
exec_dir = "/".join(current_dir.split("/")[:-1])
exec_path = os.path.join(exec_dir, "exec.py")

base_command = '''bsub \\
-gpu num=1:j_exclusive=yes:mode=exclusive_process:gmem=10.7G \\
-L /bin/bash -q gpu-lowprio \\
-u 'till.bungert@dkfz-heidelberg.de' -B -N \\
"source ~/.bashrc && conda activate $CONDA_ENV/failure-detection && python -W ignore {} {}"'''
# base_command = "EXPERIMENT_ROOT_DIR=~/cluster/experiments DATASET_ROOT_DIR=~/Data python -W ignore {} {}"

datasets = ["cifar10", "cifar100"]
runs = range(1, 5)
lrs = [0.0003, 0.01]
for run, (dataset, lr) in product(runs, zip(datasets, lrs)):
    command_line_args = ""
    command_line_args += "study={}_vit_study ".format(dataset)
    command_line_args += "exp.name={}_lr{}_run{} ".format(dataset, lr, run)
    command_line_args += "exp.mode={} ".format("test")
    # command_line_args += "exp.mode={} ".format("analysis")
    command_line_args += "trainer.learning_rate={} ".format(lr)
    command_line_args += "+trainer.do_val=true "
    command_line_args += "+eval.val_tuning=false "
    command_line_args += "trainer.batch_size=128 "

    launch_command = base_command.format(exec_path, command_line_args)

    print("Launch command: ", launch_command)
    subprocess.call(launch_command, shell=True)
    time.sleep(1)
