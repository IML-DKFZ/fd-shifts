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
"source ~/.bashrc && conda activate $CONDA_ENV/failure-detection && python -W ignore {} {}"'''
# base_command = "EXPERIMENT_ROOT_DIR=~/cluster/experiments DATASET_ROOT_DIR=~/Data python -W ignore {} {}"

mode = "train" # "test" / "train"
# backbones = ["vgg13", "vgg16"]
backbones = ["vit"]
dropouts = [0] # only true for vgg16
models = ["devries_model"]
runs = [0]
# scheduler = ["MultiStep", "CosineAnnealing"]
scheduler = ["CosineAnnealing"]
avg_pool = [True]
num_epochs = [250]
norms = ["orig"] #

for ix, (bb, do, model, run, ne, ap, sched, norm) in enumerate(product(backbones, dropouts, models, runs, num_epochs, avg_pool, scheduler, norms)):

    exp_group_name = "devries_vit"
    exp_name = "{}_bb{}_do{}_run{}_ne{}_ap{}_{}_norm{}".format(model, bb, do, run, ne, ap, sched, norm)
    command_line_args = ""

    if mode == "test":
        command_line_args += "--config-path=$EXPERIMENT_ROOT_DIR/{} ".format(os.path.join(exp_group_name, exp_name, "hydra"))
        command_line_args += "exp.mode=test "
        # command_line_args += "eval.query_studies.new_class_study=\"{}\" ".format(
        #     ['tinyimagenet', 'tinyimagenet_resize', "cifar100", "svhn"])
    else:
        command_line_args += "study={} ".format("cifar_devries_study")
        command_line_args += "data={} ".format("cifar10_384_data")
        # command_line_args += "exp.group_name={} ".format(exp_group_name)
        command_line_args += "exp.name={} ".format(exp_name)
        command_line_args += "exp.mode={} ".format("train_test")
        command_line_args += "trainer.num_epochs={} ".format(ne)
        command_line_args += "trainer.lr_scheduler.name={} ".format(sched)
        command_line_args += "trainer.batch_size=128 "
        command_line_args += "+trainer.accelerator=dp "

        command_line_args += "model.dropout_rate={} ".format(do)
        command_line_args += "model.name={} ".format(model) # todo careful, name vs backbone!
        command_line_args += "model.network.name={} ".format(
            "devries_and_enc")  # todo careful, name vs backbone!
        command_line_args += "model.network.backbone={} ".format(bb)  # todo careful, name vs backbone!
        command_line_args += "eval.confidence_measures.test=\"{}\" ".format(["det_mcp" , "det_pe", "ext"])

        # if do == 1:
        # command_line_args += "model.avg_pool={} ".format(ap)
        # else:
        #     command_line_args += "model.network.name={} ".format(bb)
        if norm == "devries":
            command_line_args += "data.augmentations.train.normalize=\"{}\" ".format([[0.4913725490196078, 0.4823529411764706, 0.4466666666666667], [0.24705882352941178, 0.24352941176470588, 0.2615686274509804]])
            command_line_args += "data.augmentations.val.normalize=\"{}\" ".format([[0.4913725490196078, 0.4823529411764706, 0.4466666666666667], [0.24705882352941178, 0.24352941176470588, 0.2615686274509804]])
            command_line_args += "data.augmentations.test.normalize=\"{}\" ".format([[0.4913725490196078, 0.4823529411764706, 0.4466666666666667], [0.24705882352941178, 0.24352941176470588, 0.2615686274509804]])
            command_line_args += "test.assim_ood_norm_flag=True "

        if do == 1:
            command_line_args += "eval.confidence_measures.test=\"{}\" ".format(
                ["det_mcp", "det_pe", "ext", "ext_mcd", "ext_waic", "mcd_mcp", "mcd_pe", "mcd_ee", "mcd_mi",
                 "mcd_sv", "mcd_waic"])
        # else:
        #     command_line_args += "eval.confidence_measures.test=\"{}\" ".format(
        #         ["det_mcp", "det_pe"])

    launch_command = base_command.format(exec_path, command_line_args)

    print("Launch command: ", launch_command)
    subprocess.call(launch_command, shell=True)
    time.sleep(1)
