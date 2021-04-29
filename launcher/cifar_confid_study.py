import os
import subprocess
from itertools import product

system_name = os.environ['SYSTEM_NAME']
# sur_out_name = os.path.join(exp_group_dir, "surveillance_sheet.txt")

current_dir = os.path.dirname(os.path.realpath(__file__))
exec_dir = "/".join(current_dir.split("/")[:-1])
exec_path = os.path.join(exec_dir,"exec.py")

datasets = ["cifar10", "cifar100"]
pt_paths = ["/gpu/checkpoints/OE0612/jaegerp/repro_cifar_confid/cifar10_run_0_sebastian/version_1/best_valacc.ckpt",
            "/gpu/checkpoints/OE0612/jaegerp/repro_cifar_confid/cifar100_run_1_sebastian/version_1/best_valacc.ckpt"
            ]
runs = range(2)
repro_mode = [True, True]
for run, rm, dataset, pretrained_path in zip(runs, repro_mode, datasets, pt_paths):
    if run == 0:
        command_line_args = ""
        command_line_args += "study={} ".format("cifar_confid_study")
        command_line_args += "data={} ".format("{}_data".format(dataset))
        command_line_args += "exp.group_name={} ".format("repro_cifar_confid")
        command_line_args += "exp.name={} ".format("{}_run_{}_annealconfidFIXsmall".format(dataset,run))
        if rm:
            command_line_args += "data.reproduce_confidnet_splits={} ".format("True")
        # if fold>0:
        #     command_line_args += "exp.fold={} ".format(fold)
        command_line_args += "exp.mode={} ".format("train_test")
        # command_line_args += "trainer.weight_decay={} ".format(wd)
        # command_line_args += "trainer.multistep_lr_milestones=\"{}\" ".format(ms)
        # command_line_args += "model.network.imagenet_weights_path=null"
        command_line_args += "trainer.callbacks.training_stages.pretrained_backbone_path={}".format(pretrained_path)


        if system_name == "cluster":

            launch_command = ""
            launch_command += "bsub "
            launch_command += "-gpu num=1:"
            launch_command += "j_exclusive=yes:"
            launch_command += "mode=exclusive_process:"
            launch_command += "gmem=10.7G "
            launch_command += "-L /bin/bash -q gpu "
            launch_command += "-u 'p.jaeger@dkfz-heidelberg.de' -B -N "
            launch_command += "'source ~/.bashrc && "
            launch_command += "source ~/.virtualenvs/confid/bin/activate && "
            launch_command += "python -u {} ".format(exec_path)
            launch_command += command_line_args
            launch_command += "'"

        elif system_name == "mbi":
            launch_command = "python -u {} ".format(exec_path)
            launch_command += command_line_args

        else:
            RuntimeError("system_name environment variable not known.")

        print("Launch command: ", launch_command)
        subprocess.call(launch_command, shell=True)

#subprocess.call("python {}/utils/job_surveillance.py --in_name {} --out_name {} --n_jobs {} &".format(exec_dir, log_folder, sur_out_name, job_ix), shell=True)