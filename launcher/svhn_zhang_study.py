import os
import subprocess
from itertools import product
import time

system_name = os.environ['SYSTEM_NAME']
# sur_out_name = os.path.join(exp_group_dir, "surveillance_sheet.txt")

current_dir = os.path.dirname(os.path.realpath(__file__))
exec_dir = "/".join(current_dir.split("/")[:-1])
exec_path = os.path.join(exec_dir,"exec.py")


noises = [0, 1e-4]
wds = [0.0005]
bss = [64]
dos = [0]
for ix, (noise, wd, bs, do) in enumerate(product(noises, wds, bss, dos)):

    # if ix < 2:

        command_line_args = ""
        command_line_args += "study={} ".format("svhn_zhang_study")
        command_line_args += "data={} ".format("svhn_data")
        command_line_args += "exp.group_name={} ".format("zhang_svhn")
        command_line_args += "exp.name={} ".format("zhang_svhn_noise{}_wd{}_bs{}_do{}".format(noise, wd, bs, do))
        command_line_args += "data.reproduce_confidnet_splits={} ".format("True")
        # if fold>0:
        #     command_line_args += "exp.fold={} ".format(fold)
        command_line_args += "exp.mode={} ".format("train_test")

        command_line_args += "model.epsilon_noise={} ".format(noise)
        command_line_args += "trainer.weight_decay.flow={} ".format(wd)
        command_line_args += "trainer.batch_size={} ".format(bs)
        command_line_args += "model.dropout_rate={} ".format(do)


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