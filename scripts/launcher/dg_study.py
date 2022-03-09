import os
import subprocess
from itertools import product
import time


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

mode = "train" # "test" / "train"
# backbones = ["vgg13", "vgg16"]
backbones = ["vit"]
dropouts = [1, 0]
cutout = [True]# only true for vgg16
models = ["devries_model"]
scheduler = ["LinearWarmupCosineAnnealing"]
reward = [2.2, 3, 4.5, 6, 10]
runs = [1]
lrs = [1e-1, 1e-3]


for ix, (bb, do, model, run, rew, sched, co, lr) in enumerate(product(backbones, dropouts, models, runs, reward, scheduler, cutout, lrs)):

    if not (do == 0 and co == False) and (do==1):
        exp_group_name = "dg_sweep_ultimate"
        exp_name = "{}_bb{}_do{}_run{}_rew{}_sched{}_co{}_lr{}".format(model, bb, do, run, rew, sched, co, lr)
        command_line_args = ""

        if mode == "test":
            command_line_args += "--config-path=$EXPERIMENT_ROOT_DIR/{} ".format(os.path.join(exp_group_name, exp_name, "hydra"))
            command_line_args += "exp.mode=test "
            command_line_args += "eval.confidence_measures.test=\"{}\" ".format(
                ["det_mcp", "det_pe", "mcd_mcp", "mcd_pe", "mcd_ee", "mcd_mi", "mcd_sv", "ext","mcd_waic", "ext_waic", "ext_mcd"])
        elif mode == "analysis":
            command_line_args += "--config-path=$EXPERIMENT_ROOT_DIR/{} ".format(
                os.path.join(exp_group_name, exp_name, "hydra"))
            command_line_args += "exp.mode=analysis "
            command_line_args += "eval.confidence_measures.test=\"{}\" ".format(
                ["det_mcp", "det_pe", "mcd_mcp", "mcd_pe", "mcd_ee", "mcd_mi", "mcd_sv", "ext", "mcd_waic", "ext_waic", "ext_mcd"])
        else:
            command_line_args += "study={} ".format("dg_cifar_study")
            command_line_args += "data={} ".format("cifar10_384_data")
            # command_line_args += "exp.group_name={} ".format(exp_group_name)
            command_line_args += "exp.name={} ".format(exp_name)
            command_line_args += "exp.mode={} ".format("train_test")
            command_line_args += "trainer.num_epochs={} ".format(300)
            command_line_args += "trainer.dg_pretrain_epochs={} ".format(100)
            command_line_args += "trainer.lr_scheduler.name={} ".format(sched)
            command_line_args += "model.dg_reward={} ".format(rew)
            command_line_args += "trainer.batch_size=128 "
            command_line_args += "+trainer.accelerator=dp "
            command_line_args += "trainer.optimizer.learning_rate={} ".format(lr)

            command_line_args += "model.dropout_rate={} ".format(do)
            command_line_args += "model.name={} ".format(model) # todo careful, name vs backbone!
            command_line_args += "model.network.name={} ".format(bb)  # todo careful, name vs backbone!
            command_line_args += "model.network.backbone={} ".format("null")  # todo careful, name vs backbone!
            if model != "det_mcd_model":
                command_line_args += "eval.confidence_measures.test=\"{}\" ".format(["det_mcp" , "det_pe", "ext"])
                command_line_args += "eval.ext_confid_name=\"{}\" ".format("dg")

            if do == 1:
              command_line_args += "model.avg_pool={} ".format(False)

            if co == False:
                command_line_args += "~data.augmentations.train.cutout "

    launch_command = base_command.format(exec_path, command_line_args)

    print("Launch command: ", launch_command)
    subprocess.call(launch_command, shell=True)
    time.sleep(1)
