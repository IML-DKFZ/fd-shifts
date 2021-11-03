import os
import subprocess
import time
from itertools import product

current_dir = os.path.dirname(os.path.realpath(__file__))
exec_dir = "/".join(current_dir.split("/")[:-1])
exec_path = os.path.join(exec_dir, "exec.py")

base_command = '''bsub \\
-R "select[hname!='e230-dgx2-1']" \\
-gpu num=4:j_exclusive=yes:mode=exclusive_process:gmem=31.7G \\
-L /bin/bash -q gpu-lowprio \\
-u 'till.bungert@dkfz-heidelberg.de' -B -N \\
"source ~/.bashrc && conda activate $CONDA_ENV/failure-detection && python -W ignore {} {}"'''

datasets = ["cifar10", "cifar100", "svhn"]
runs = range(1)
lrs = [0.0001, 0.0003, 0.001, 0.003]
for run, dataset, lr in product(runs, datasets, lrs):
    command_line_args = ""
    command_line_args += "study={}_vit_study ".format(dataset)
    command_line_args += "exp.name={}_lr{}_run{} ".format(dataset, lr, run)
    command_line_args += "exp.mode={} ".format("train")
    command_line_args += "trainer.learning_rate={} ".format(lr)
    command_line_args += "+trainer.do_val=true "

    launch_command = base_command.format(exec_path, command_line_args)

    print("Launch command: ", launch_command)
    subprocess.call(launch_command, shell=True)
    time.sleep(1)
