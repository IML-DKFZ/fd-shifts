import os
import subprocess
from itertools import product
import time

system_name = os.environ['SYSTEM_NAME']
# sur_out_name = os.path.join(exp_group_dir, "surveillance_sheet.txt")

current_dir = os.path.dirname(os.path.realpath(__file__))
exec_dir = "/".join(current_dir.split("/")[:-1])
exec_path = os.path.join(exec_dir,"exec.py")

mode = "test" # "test" / "train"
backbones = ["vgg16"]
dropouts = [False, True] # only true for vgg16
assim_norms = [False, True]
schedulers = ["CosineAnnealing", "MultiStep"]

for ix, (bb, do, norm, sched) in enumerate(product(backbones, dropouts, assim_norms, schedulers)):

    if not (bb == "vgg13" and do == True):

        exp_group_name = "devries_sweep"
        exp_name = "det_devries_bb{}_do{}_norm{}_sched{}".format(bb, do, norm, sched)
        command_line_args = ""

        if mode == "test":
            command_line_args += "--config-path=$EXPERIMENT_ROOT_DIR/{} ".format(os.path.join(exp_group_name, exp_name, "hydra"))
            command_line_args += "exp.mode=test "
        else:
            command_line_args += "study={} ".format("cifar_devries_study")
            command_line_args += "data={} ".format("cifar10_data")
            command_line_args += "exp.group_name={} ".format(exp_group_name)
            command_line_args += "exp.name={} ".format(exp_name)
            command_line_args += "exp.mode={} ".format("train_test")

            command_line_args += "model.network.name={} ".format(bb) # todo careful, name vs backbone!
            command_line_args += "model.dropout_flag={} ".format(do)
            command_line_args += "trainer.lr_scheduler.name={} ".format(sched)
            command_line_args += "test.assim_ood_norm_flag={} ".format(norm)


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
        time.sleep(1)

#subprocess.call("python {}/utils/job_surveillance.py --in_name {} --out_name {} --n_jobs {} &".format(exec_dir, log_folder, sur_out_name, job_ix), shell=True)