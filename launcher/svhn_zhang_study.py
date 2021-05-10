import os
import subprocess
from itertools import product
import time

system_name = os.environ['SYSTEM_NAME']
# sur_out_name = os.path.join(exp_group_dir, "surveillance_sheet.txt")

current_dir = os.path.dirname(os.path.realpath(__file__))
exec_dir = "/".join(current_dir.split("/")[:-1])
exec_path = os.path.join(exec_dir,"exec.py")

mode = "train"
batch_sizes = [128]
wds = [0.0005]
wdcs = [0.0005]
backbones = ["vgg13", "svhn_small_conv"] #  "zhang_backbone",
scheds = [True] # only flow scheduler!!
num_epochs = [200]
dropout = [0]
cutouts = [True]


for ix, (bs, wd, wdc, bb, sched, ne, do, cutout) in enumerate(product(batch_sizes, wds, wdcs, backbones, scheds, num_epochs, dropout, cutouts)):

    # if ix < 2:
    exp_group_name = "zhang_realcifar_sweep"
    exp_name = "zhang_bb{}_wd{}_wdc{}_bs{}_scehd{}_ne{}_do{}_cutout{}".format(bs, wd, wdc, bb, sched, ne, do, cutout)
    command_line_args = ""

    if mode == "analysis":
        command_line_args += "--config-path=$EXPERIMENT_ROOT_DIR/{} ".format(
            os.path.join(exp_group_name, exp_name, "hydra"))
        command_line_args += "exp.mode=analysis "
        command_line_args += "+eval.val_tuning=False "

    else:

        command_line_args += "study={} ".format("svhn_zhang_study")
        command_line_args += "data={} ".format("cifar10_data")
        command_line_args += "exp.group_name={} ".format(exp_group_name)
        command_line_args += "exp.name={} ".format(exp_name)

        # command_line_args += "data.reproduce_confidnet_splits={} ".format("True")
        # if fold>0:
        #     command_line_args += "exp.fold={} ".format(fold)
        command_line_args += "exp.mode={} ".format("train_test")

        command_line_args += "trainer.weight_decay.flow={} ".format(wd)
        command_line_args += "trainer.weight_decay.classifier={} ".format(wdc)
        command_line_args += "trainer.batch_size={} ".format(bs)
        command_line_args += "model.network.backbone={} ".format(bb)
        command_line_args += "model.dropout_rate={} ".format(do)
        command_line_args += "trainer.lr_scheduler.apply={} ".format(sched)
        command_line_args += "trainer.num_epochs={} ".format(ne)

        # if cutout == True:
        #     command_line_args += "+data.augmentations.train.cutout=16 "

        if bs == 64:
            command_line_args += "trainer.learning_rate.classifier={} ".format(0.01)
            command_line_args += "trainer.learning_rate.flow={} ".format(0.0001)
        if ne > 60:
            command_line_args += "trainer.learning_rate.classifier={} ".format(0.001)
            command_line_args += "trainer.learning_rate.flow={} ".format(0.000001)


    if system_name == "cluster":

        launch_command = ""
        launch_command += "bsub "
        launch_command += "-gpu num=1:"
        launch_command += "j_exclusive=yes:"
        launch_command += "mode=exclusive_process:"
        launch_command += "gmem=10.7G "
        launch_command += "-L /bin/bash -q gpu-lowprio "
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