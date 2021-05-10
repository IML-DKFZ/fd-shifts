import os
import subprocess
from itertools import product
import time

system_name = os.environ['SYSTEM_NAME']
# sur_out_name = os.path.join(exp_group_dir, "surveillance_sheet.txt")

current_dir = os.path.dirname(os.path.realpath(__file__))
exec_dir = "/".join(current_dir.split("/")[:-1])
exec_path = os.path.join(exec_dir,"exec.py")



mode = "test" # "test" / "train" / "analysis"
backbones = ["vgg13","vgg16"] #
dropouts = [1, 0] # #
models = ["confidnet_model"]
num_epochs = [200, 250]
runs = [0, 1 , 2]
avg_pool = [False, True]
my_ix = 0

for ix, (bb, do, model, ne, run, ap) in enumerate(product(backbones, dropouts, models, num_epochs, runs, avg_pool)):

    if ix == 0:
    # if not (model == "det_mcd_model" and do == 0) and not (model!="confidnet_model" and ms==False) and not (model=="devries_model" and do==1) and model == "confidnet_model":
        my_ix += 1
        exp_group_name = "tcp_decision_sweep_latest"
        # exp_name = "{}_bb{}_do{}_avgpoolTrue_ms{}".format(model, bb, do, ms)
        exp_name = "{}_bb{}_do{}_ne{}_run{}_ap{}".format(model, bb, do, ne, run, ap)
        print(exp_name)
        command_line_args = ""

        if mode == "test":
            command_line_args += "--config-path=$EXPERIMENT_ROOT_DIR/{} ".format(os.path.join(exp_group_name, exp_name, "hydra"))
            command_line_args += "exp.mode=test "
            command_line_args += "trainer.num_epochs=250 "
            command_line_args += "trainer.num_epochs_backbone=0 "
            milestones = [0, 200]
            backbone_path = "$EXPERIMENT_ROOT_DIR/{} ".format(os.path.join("bak_tcp_decision_sweep", exp_name, "version_0", "last.ckpt"))
            command_line_args += "trainer.callbacks.training_stages.milestones=\"{}\" ".format(milestones)
            command_line_args += "trainer.callbacks.training_stages.pretrained_backbone_path={} ".format(backbone_path)

            # TODO MODEL SELECTION!!
            # command_line_args += "test.selection_criterion=latest " # todo if not ms!!

            # command_line_args += " +exp.output_paths={} "
            # command_line_args += " +exp.output_paths.test={} "
            # command_line_args += " +exp.output_paths.test.raw_output=$EXPERIMENT_ROOT_DIR/{} ".format(os.path.join(exp_group_name, exp_name, "test_results", "raw_output.npy"))
            # command_line_args += " +exp.output_paths.test.raw_output_dist=$EXPERIMENT_ROOT_DIR/{} ".format(os.path.join(exp_group_name, exp_name, "test_results", "raw_output_dist.npy"))
            # command_line_args += " +exp.output_paths.test.external_confids=$EXPERIMENT_ROOT_DIR/{} ".format(os.path.join(exp_group_name, exp_name, "test_results", "external_confids.npy"))
            # command_line_args += " +exp.output_paths.test.external_confids_dist=$EXPERIMENT_ROOT_DIR/{} ".format(os.path.join(exp_group_name, exp_name, "test_results", "external_confids_dist.npy"))
            # command_line_args += " +exp.output_paths.test.raw_output_dist: ${test.dir} / raw_output_dist.npy "
            # command_line_args += " +exp.output_paths.test.external_confids: ${test.dir} / external_confids.npy "
            # command_line_args += " +exp.output_paths.test.external_confids_dist: ${test.dir} / external_confids_dist.npy "


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
                command_line_args += "trainer.num_epochs_backbone={} ".format(ne)
                num_epochs = ne + 220
                milestones = [ne, ne + 200]
                command_line_args += "trainer.num_epochs={} ".format(num_epochs)
                command_line_args += "trainer.callbacks.training_stages.milestones=\"{}\" ".format(milestones)

            else:
                command_line_args += "study={} ".format("cifar_devries_study")
                command_line_args += "model.network.name={} ".format(bb)
                command_line_args += "model.trainer.num_epochs={} ".format(ne)

            command_line_args += "data={} ".format("cifar10_data")
            command_line_args += "exp.group_name={} ".format(exp_group_name)
            command_line_args += "exp.name={} ".format(exp_name)
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

print(my_ix)
#subprocess.call("python {}/utils/job_surveillance.py --in_name {} --out_name {} --n_jobs {} &".format(exec_dir, log_folder, sur_out_name, job_ix), shell=True)