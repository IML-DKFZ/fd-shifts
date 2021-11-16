import os
import subprocess
import time
from itertools import product

current_dir = os.path.dirname(os.path.realpath(__file__))
exec_dir = "/".join(current_dir.split("/")[:-1])
exec_path = os.path.join(exec_dir, "exec.py")

# datasets = ["svhn", "breeds", "wilds_camelyon", "cifar100", "wilds_animals"]
# lrs = [0.01, 1e-3, 0.001, 0.003, 0.0001]
# dos = [0]
# runs = range(1, 5)
# for run, (dataset, lr), do in product(runs, zip(datasets, lrs), dos):
experiments = [
    ["cifar10", 0.01, 1, range(1, 5)],
    ["cifar100", 0.03, 0, range(1, 5)],
    ["super_cifar100", 0.003, 0, range(1, 5)],
    ["super_cifar100", 0.001, 1, range(1, 5)],
    ["breeds", 0.01, 1, range(1, 5)],
    ["svhn", 0.01, 1, range(1, 5)],
    ["wilds_animals", 0.01, 1, range(1, 5)],
    ["wilds_animals", 0.001, 0, [4]],
    ["wilds_camelyon", 0.003, 1, range(1, 5)],
]
for dataset, lr, do, runs in experiments:
    for run in runs:
        if dataset in ["cifar100", "wilds_animals"]:
            base_command = '''bsub \\
            -R "select[hname!='e230-dgx2-1']" \\
            -gpu num=16:j_exclusive=yes:mode=exclusive_process:gmem=31.7G \\
            -L /bin/bash -q gpu-lowprio \\
            -u 'till.bungert@dkfz-heidelberg.de' -B -N \\
            -J "{}_lr{}_run{}_do{}" \\
            "source ~/.bashrc && conda activate $CONDA_ENV/failure-detection && python -W ignore {} {}"'''
        else:
            base_command = '''bsub \\
            -R "select[hname!='e230-dgx2-1']" \\
            -gpu num=4:j_exclusive=yes:mode=exclusive_process:gmem=31.7G \\
            -L /bin/bash -q gpu-lowprio \\
            -u 'till.bungert@dkfz-heidelberg.de' -B -N \\
            -J "{}_lr{}_run{}_do{}" \\
            "source ~/.bashrc && conda activate $CONDA_ENV/failure-detection && python -W ignore {} {}"'''


        command_line_args = ""
        command_line_args += "study={}_vit_study ".format(dataset)
        command_line_args += "exp.name={}_lr{}_run{}_do{} ".format(dataset, lr, run, do)
        command_line_args += "exp.mode={} ".format("train")
        command_line_args += "trainer.learning_rate={} ".format(lr)
        command_line_args += "trainer.val_split=devries "
        command_line_args += "+trainer.do_val=true "
        command_line_args += "+eval.val_tuning=true "
        command_line_args += "+eval.r_star=0.25 "
        command_line_args += "+eval.r_delta=0.05 "
        command_line_args += "+model.dropout_rate={} ".format(do)
        command_line_args += "+trainer.accelerator=dp "

        launch_command = base_command.format(dataset, lr, run, do, exec_path, command_line_args)

        print("Launch command: ", launch_command)
        subprocess.call(launch_command, shell=True)
        time.sleep(1)

        # TESTING

        base_command = '''bsub \\
        -gpu num=4:j_exclusive=yes:gmem=10.7G \\
        -L /bin/bash -q gpu-lowprio \\
        -u 'till.bungert@dkfz-heidelberg.de' -B -N \\
        -w "done({}_lr{}_run{}_do{})" \\
        "source ~/.bashrc && conda activate $CONDA_ENV/failure-detection && python -W ignore {} {}"'''

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

        launch_command = base_command.format(dataset, lr, run, do, exec_path, command_line_args)

        print("Launch command: ", launch_command)
        subprocess.call(launch_command, shell=True)
        time.sleep(1)
