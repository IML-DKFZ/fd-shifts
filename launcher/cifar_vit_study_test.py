import os
import subprocess
import time
from itertools import product

current_dir = os.path.dirname(os.path.realpath(__file__))
exec_dir = "/".join(current_dir.split("/")[:-1])
exec_path = os.path.join(exec_dir, "exec.py")


datasets = ["cifar100", "wilds_animals"]
lrs = [0.01, 0.03, 0.001, 0.003]
dos = [1]
runs = range(1)
for run, dataset, lr, do in product(runs, datasets, lrs, dos):
    base_command = '''bsub \\
    -gpu num=4:j_exclusive=yes:gmem=10.7G \\
    -R "select[hname!='e230-dgx2-1']" \\
    -L /bin/bash -q gpu-lowprio \\
    -u 'till.bungert@dkfz-heidelberg.de' -B -N \\
    'source ~/.bashrc && conda activate $CONDA_ENV/failure-detection && python -W ignore {} {}\''''
    # base_command = "EXPERIMENT_ROOT_DIR=~/cluster/experiments DATASET_ROOT_DIR=~/Data python -W ignore {} {}"

    command_line_args = ""
    command_line_args += "study={}_vit_study ".format(dataset)
    command_line_args += "exp.name={}_lr{}_run{}_do{} ".format(dataset, lr, run, do)
    command_line_args += "exp.mode={} ".format("test")
    command_line_args += "trainer.learning_rate={} ".format(lr)
    command_line_args += "trainer.val_split=devries "
    command_line_args += "+trainer.do_val=true "
    command_line_args += "+eval.val_tuning=true "
    command_line_args += "+model.dropout_rate={} ".format(do)
    command_line_args += "+eval.r_star=0.25 "
    command_line_args += "+eval.r_delta=0.05 "
    command_line_args += "trainer.batch_size=128 "
    command_line_args += "+trainer.accelerator=ddp "

    if do == 1:
        command_line_args += "eval.confidence_measures.test=\"{}\" ".format(
            ["det_mcp" , "det_pe", "ext", "ext_mcd", "ext_waic", "mcd_mcp", "mcd_pe", "mcd_ee", "mcd_mi", "mcd_sv", "mcd_waic"])

    launch_command = base_command.format(exec_path, command_line_args)

    print("Launch command: ", launch_command)
    subprocess.call(launch_command, shell=True)
    time.sleep(1)
