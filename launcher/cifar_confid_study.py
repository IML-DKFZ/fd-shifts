import os
import subprocess
from itertools import product


current_dir = os.path.dirname(os.path.realpath(__file__))
exec_dir = "/".join(current_dir.split("/")[:-1])
exec_path = os.path.join(exec_dir,"exec.py")

base_command = '''bsub \\
-R "select[hname!='e230-dgx2-1']" \\
-gpu num=4:j_exclusive=yes:mode=exclusive_process:gmem=31.7G \\
-L /bin/bash -q gpu-lowprio \\
-u 'till.bungert@dkfz-heidelberg.de' -B -N \\
'source ~/.bashrc && conda activate $CONDA_ENV/failure-detection && python -W ignore {} {}\''''
# base_command = "EXPERIMENT_ROOT_DIR=~/cluster/experiments DATASET_ROOT_DIR=~/Data python -W ignore {} {}"

datasets = ["cifar10"]
pt_paths = [
    "/home/t974t/cluster/experiments/vit/cifar10_lr0.0003_run0/version_0/last.ckpt",
]
runs = range(1)
repro_mode = [True]
for run, rm, dataset, pretrained_path in zip(runs, repro_mode, datasets, pt_paths):
    if run == 0:
        command_line_args = ""
        command_line_args += "study={} ".format("cifar_tcp_confid_sweep")
        command_line_args += "data={} ".format("{}_384_data".format(dataset))
        # command_line_args += "exp.group_name={} ".format("repro_cifar_confid")
        command_line_args += "exp.name={} ".format("{}_run{}_annealconfidFIXsmall".format(dataset, run))
        if rm:
            command_line_args += "data.reproduce_confidnet_splits={} ".format("True")
        # if fold>0:
        #     command_line_args += "exp.fold={} ".format(fold)
        command_line_args += "exp.mode={} ".format("train_test")
        command_line_args += "trainer.callbacks.training_stages.pretrained_backbone_path={} ".format(pretrained_path)
        command_line_args += "model.network.backbone=vit "
        command_line_args += "+trainer.accelerator=dp "

        launch_command = base_command.format(exec_path, command_line_args)

        print("Launch command: ", launch_command)
        subprocess.call(launch_command, shell=True)

#subprocess.call("python {}/utils/job_surveillance.py --in_name {} --out_name {} --n_jobs {} &".format(exec_dir, log_folder, sur_out_name, job_ix), shell=True)
