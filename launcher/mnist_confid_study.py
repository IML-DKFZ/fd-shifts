import os
import subprocess
from itertools import product

system_name = os.environ['SYSTEM_NAME']
# sur_out_name = os.path.join(exp_group_dir, "surveillance_sheet.txt")

current_dir = os.path.dirname(os.path.realpath(__file__))
exec_dir = "/".join(current_dir.split("/")[:-1])
exec_path = os.path.join(exec_dir,"exec.py")

runs = range(8)
repro_mode = [True, True, True, True, False, False, False, False]
disable_dropout = [True, False, True, False, True, False, True, False]
folds = [0, 0, 0, 0, 1, 1, 2, 2]

for model in ["mnist_mlp", "mnist_small_conv"]:

    for run, rm, fold, dd in zip(runs, repro_mode, folds, disable_dropout):

        if not (model=="mnist_mlp" and fold>0):

            command_line_args = ""
            command_line_args += "study={} ".format("mnist_confid_study")
            command_line_args += "data={} ".format("mnist_data")
            command_line_args += "exp.name={} ".format("repro_confid_svhn_run_{}_fold_{}_rm_{}_dd_{}_{}".format(run, fold, "yes" if rm else "no", "yes" if dd else "no", model))
            command_line_args += "exp.group_name={} ".format("repro_mnist")
            if rm:
                command_line_args += "data.reproduce_confidnet_splits={} ".format("True")
            if dd:
                command_line_args += "trainer.callbacks.training_stages.disable_dropout_at_finetuning={} ".format("True")
            if fold>0:
                command_line_args += "exp.fold={} ".format(fold)
            command_line_args += "exp.mode={} ".format("train_test")
            command_line_args += "model.fc_dim={} ".format(128 if model == "mnist_small_conv" else 1000) # 1000 for mlp
            command_line_args += "model.network.backbone={} ".format(model) # "mnist_mlp"

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