import os
import subprocess
from itertools import product
import time

system_name = os.environ['SYSTEM_NAME']
# sur_out_name = os.path.join(exp_group_dir, "surveillance_sheet.txt")

current_dir = os.path.dirname(os.path.realpath(__file__))
exec_dir = "/".join(current_dir.split("/")[:-1])
exec_path = os.path.join(exec_dir,"exec.py")



mode = "train_confidnet" # "test" / "train" / "analysis"
backbones = ["vgg13","vgg16"] #
dropouts = [1, 0] # #
models = ["confidnet_model"]
num_epochs = [200, 250]
runs = [0, 1 , 2]
avg_pool = [False, True]
my_ix = 0
tune_lr = [1e-6] # 1e-7
model_selection = ["best_failap_err"] # best_failap_err

for ix, (bb, do, model, ne, run, ap, tlr, ms) in enumerate(product(backbones, dropouts, models, num_epochs, runs, avg_pool, tune_lr, model_selection)):

    if  18 <= ix:
    # if not (model == "det_mcd_model" and do == 0) and not (model!="confidnet_model" and ms==False) and not (model=="devries_model" and do==1) and model == "confidnet_model":
        my_ix += 1
        exp_group_name = "tcp_decision_sweep"
        # exp_name = "{}_bb{}_do{}_avgpoolTrue_ms{}".format(model, bb, do, ms)
        old_exp_name = "{}_bb{}_do{}_ne{}_run{}_ap{}".format(model, bb, do, ne, run, ap)
        new_exp_name = "{}_bb{}_do{}_ne{}_run{}_ap{}_{}tlr_{}ms".format(model, bb, do, ne, run, ap, tlr, ms)

        command_line_args = ""

        if mode == "test":
            command_line_args += "--config-path=$EXPERIMENT_ROOT_DIR/{} ".format(os.path.join(exp_group_name, exp_name, "hydra"))
            command_line_args += "exp.mode=test "


        elif mode == "analysis":
            command_line_args += "--config-path=$EXPERIMENT_ROOT_DIR/{} ".format(
                os.path.join(exp_group_name, exp_name, "hydra"))
            command_line_args += "exp.mode=analysis "
            command_line_args += "+eval.query_studies.new_class_study=\"{}\" ".format(['tinyimagenet', 'tinyimagenet_resize'])

        else:

            if "devries" in model:
                command_line_args += "study={} ".format("cifar_devries_study")
                command_line_args += "eval.ext_confid_name={} ".format("devries")
                command_line_args += "model.network.name={} ".format("devries_and_enc")
                command_line_args += "trainer.num_epochs={} ".format(ne)
            elif "confid" in model:
                command_line_args += "study={} ".format("cifar_tcp_confid_sweep")
                command_line_args += "eval.ext_confid_name={} ".format("tcp")
                command_line_args += "model.network.name={} ".format("confidnet_and_enc")
                # command_line_args += "trainer.num_epochs_backbone={} ".format(ne)
                # num_epochs = ne + 220
                # milestones = [ne, ne + 200]
                # command_line_args += "trainer.num_epochs={} ".format(num_epochs)
                # command_line_args += "trainer.callbacks.training_stages.milestones=\"{}\" ".format(milestones)
                if mode == "finetune_confidnet" or mode == "train_confidnet":
                    if mode == "finetune_confidnet":
                        num_epochs = 50
                        milestones = [0, 0]
                        confidnet_path = "$EXPERIMENT_ROOT_DIR/{} ".format(
                            os.path.join("tcp_decision_sweep", old_exp_name, "version_0", "last.ckpt"))
                        backbone_path = "$EXPERIMENT_ROOT_DIR/{} ".format(
                            os.path.join("bak_tcp_decision_sweep", old_exp_name, "version_0", "last.ckpt"))

                    elif mode == "train_confidnet":
                        num_epochs = 250
                        milestones = [0, 200]
                        confidnet_path = "null"
                        backbone_path = "$EXPERIMENT_ROOT_DIR/{} ".format(
                            os.path.join("bak_tcp_decision_sweep", old_exp_name, "version_0", "last.ckpt"))

                    command_line_args += "trainer.num_epochs={} ".format(num_epochs)
                    command_line_args += "trainer.num_epochs_backbone=0 "
                    command_line_args += "trainer.callbacks.training_stages.milestones=\"{}\" ".format(milestones)
                    command_line_args += "trainer.callbacks.training_stages.pretrained_backbone_path={} ".format(
                        backbone_path)
                    command_line_args += "trainer.callbacks.training_stages.pretrained_confidnet_path={} ".format(
                        backbone_path)

                    # TODO this only because not given in previous sweep (not necessary for future sweeps)
                    model_selection_dict = {
                        "n": 2,
                        "selection_metric": ["val/accuracy", "val/ext_failap_err"],
                        "mode": ["max", "max"],
                        "filename": ["best_valacc", "best_failap_err"],  # min:lower is better, max: higher is better
                        "save_top_k": [1, 1]
                    }
                    command_line_args += "trainer.callbacks.model_checkpoint=\"{}\" ".format(model_selection_dict)

                    command_line_args += "test.selection_criterion={} ".format(ms)  # todo if not ms else best_failap_err!!
                    command_line_args += "trainer.learning_rate_confidnet_finetune={} ".format(tlr)



            else:
                command_line_args += "study={} ".format("cifar_devries_study")
                command_line_args += "model.network.name={} ".format(bb)
                command_line_args += "model.trainer.num_epochs={} ".format(ne)

            command_line_args += "data={} ".format("cifar10_data")
            command_line_args += "exp.group_name={} ".format(exp_group_name)
            command_line_args += "exp.name={} ".format(new_exp_name)
            command_line_args += "exp.mode={} ".format("train_test")
            command_line_args += "model.dropout_rate={} ".format(do)
            command_line_args += "model.name={} ".format(model)
            command_line_args += "model.network.backbone={} ".format(bb)


            # command_line_args += "model.avg_pool={} ".format(avg_pool)

            # if ms == False:
            #     command_line_args += "trainer.callbacks.model_checkpoint={} ".format("null")
            #     command_line_args += "test.selection_criterion=latest "

            if do and model != "det_mcd_model":
                command_line_args += "eval.confidence_measures.test=\"{}\" ".format(
                    ["det_mcp" , "det_pe", "ext", "ext_mcd", "ext_waic", "mcd_mcp", "mcd_pe", "mcd_ee", "mcd_mi", "mcd_sv", "mcd_waic"])
            elif do and model == "det_mcd_model":
                command_line_args += "eval.confidence_measures.test=\"{}\" ".format(
                    ["det_mcp", "det_pe", "mcd_mcp", "mcd_pe", "mcd_ee", "mcd_mi",
                     "mcd_sv", "mcd_waic"])
                command_line_args += "eval.confidence_measures.val=\"{}\" ".format(
                    ["det_mcp", "det_pe"])
            else:
                command_line_args += "eval.confidence_measures.test=\"{}\" ".format(
                    ["det_mcp", "det_pe", "ext"])
                command_line_args += "eval.confidence_measures.val=\"{}\" ".format(
                    ["det_mcp", "det_pe", "ext"])

        if system_name == "cluster":

            launch_command = ""
            launch_command += "bsub "
            launch_command += "-gpu num=1:"
            launch_command += "j_exclusive=yes:"
            launch_command += "mode=exclusive_process:"
            # launch_command += "gmodel=TITANXp:"
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

print(my_ix)
#subprocess.call("python {}/utils/job_surveillance.py --in_name {} --out_name {} --n_jobs {} &".format(exec_dir, log_folder, sur_out_name, job_ix), shell=True)