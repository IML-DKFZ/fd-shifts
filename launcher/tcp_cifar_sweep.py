import os
import subprocess
from itertools import product
import time

system_name = os.environ['SYSTEM_NAME']
# sur_out_name = os.path.join(exp_group_dir, "surveillance_sheet.txt")

current_dir = os.path.dirname(os.path.realpath(__file__))
exec_dir = "/".join(current_dir.split("/")[:-1])
exec_path = os.path.join(exec_dir,"exec.py")



mode = "analysis" # "test" / "train" / "analysis"
backbones = ["vgg13","vgg16"] #
dropouts = [1] # #
models = ["confidnet_model"]
# nesterov = [True, False] # todo
avg_pools = [True]
mss = [False, True] #

for ix, (bb, do, model) in enumerate(product(backbones, dropouts, models)):

    # if ms == False:

        exp_group_name = "dropout_ms_sweep"
        # exp_name = "{}_bb{}_do{}_avgpoolTrue_ms{}".format(model, bb, do, ms)
        exp_name = "{}_bb{}_do{}_avgpoolTrue".format(model, bb, do)
        command_line_args = ""

        if mode == "test":
            command_line_args += "--config-path=$EXPERIMENT_ROOT_DIR/{} ".format(os.path.join(exp_group_name, exp_name, "hydra"))
            command_line_args += "exp.mode=test "
            command_line_args += "test.selection_criterion=latest "

        elif mode == "analysis":
            command_line_args += "--config-path=$EXPERIMENT_ROOT_DIR/{} ".format(
                os.path.join(exp_group_name, exp_name, "hydra"))
            command_line_args += "exp.mode=analysis "
        else:
            if "devries" in model:
                command_line_args += "study={} ".format("cifar_devries_study")
                command_line_args += "model.network.name={} ".format("confidnet_and_enc")
            else:
                command_line_args += "study={} ".format("cifar_tcp_confid_sweep")

            command_line_args += "data={} ".format("cifar10_data")
            command_line_args += "exp.group_name={} ".format(exp_group_name)
            command_line_args += "exp.name={} ".format(exp_name)
            command_line_args += "exp.mode={} ".format("train_test")
            command_line_args += "model.dropout_rate={} ".format(do)
            command_line_args += "model.name={} ".format(model)
            command_line_args += "model.network.backbone={} ".format(bb)
            # command_line_args += "model.avg_pool={} ".format(avg_pool)
            command_line_args += "eval.ext_confid_name={} ".format("tcp")
            # if ms == False:
            #     command_line_args += "trainer.callbacks.model_checkpoint={} ".format("null")
            #     command_line_args += "test.selection_criterion=latest "

            if do:
                command_line_args += "eval.confidence_measures.test=\"{}\" ".format(
                    ["det_mcp" , "det_pe", "ext", "ext_mcd", "ext_waic", "mcd_mcp", "mcd_pe", "mcd_ee", "mcd_mi", "mcd_sv", "mcd_waic"])

            else:
                command_line_args += "eval.confidence_measures.test=\"{}\" ".format(
                    ["det_mcp", "det_pe", "ext"])

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