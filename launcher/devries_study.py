import os
import subprocess
from itertools import product
import time

system_name = os.environ['SYSTEM_NAME']
# sur_out_name = os.path.join(exp_group_dir, "surveillance_sheet.txt")

current_dir = os.path.dirname(os.path.realpath(__file__))
exec_dir = "/".join(current_dir.split("/")[:-1])
exec_path = os.path.join(exec_dir,"exec.py")

mode = "train" # "test" / "train"
backbones = ["vgg13", "vgg16"]
dropouts = [1] # only true for vgg16
models = ["devries_model"]
runs = [0, 1, 2]
avg_pool = [True, False]
num_epochs = [200, 250]

for ix, (bb, do, model, run, ne, ap) in enumerate(product(backbones, dropouts, models, runs, num_epochs, avg_pool)):


        exp_group_name = "devries_200fixed_sweep"
        exp_name = "{}_bb{}_do{}_run{}_ne{}_ap{}".format(model, bb, do, run, ne, ap)
        command_line_args = ""

        if mode == "test":
            command_line_args += "--config-path=$EXPERIMENT_ROOT_DIR/{} ".format(os.path.join(exp_group_name, exp_name, "hydra"))
            command_line_args += "exp.mode=test "
            # command_line_args += "eval.query_studies.new_class_study=\"{}\" ".format(
            #     ['tinyimagenet', 'tinyimagenet_resize', "cifar100", "svhn"])
        else:
            command_line_args += "study={} ".format("cifar_devries_study")
            command_line_args += "data={} ".format("cifar10_data")
            command_line_args += "exp.group_name={} ".format(exp_group_name)
            command_line_args += "exp.name={} ".format(exp_name)
            command_line_args += "exp.mode={} ".format("train_test")
            command_line_args += "trainer.num_epochs={} ".format(ne)

            command_line_args += "model.dropout_rate={} ".format(do)
            command_line_args += "model.name={} ".format(model) # todo careful, name vs backbone!
            command_line_args += "model.network.name={} ".format(
                "devries_and_enc")  # todo careful, name vs backbone!
            command_line_args += "model.network.backbone={} ".format(bb)  # todo careful, name vs backbone!
            command_line_args += "eval.confidence_measures.test=\"{}\" ".format(["det_mcp" , "det_pe", "ext"])

            # if do == 1:
            command_line_args += "model.avg_pool={} ".format(ap)
            # else:
            #     command_line_args += "model.network.name={} ".format(bb)

            if do == 1:
                command_line_args += "eval.confidence_measures.test=\"{}\" ".format(
                    ["det_mcp", "det_pe", "ext", "ext_mcd", "ext_waic", "mcd_mcp", "mcd_pe", "mcd_ee", "mcd_mi",
                     "mcd_sv", "mcd_waic"])
            # else:
            #     command_line_args += "eval.confidence_measures.test=\"{}\" ".format(
            #         ["det_mcp", "det_pe"])

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