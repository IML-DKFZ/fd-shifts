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
backbones = ["vgg13", "vgg16"] #
dropouts = [1] # #
models = ["confidnet_model"]
# nesterov = [True, False] #
# avg_pools = [True]
cutouts = [True]
# mss = [False, True] #
expnames = ["gobackseb_bbvgg13_fixeddo" ,    "gobackseb_bbvgg16" ,         "gobackseb_bbvgg16_totalrepro" ,
"gobackseb_bbvgg13",    "gobackseb_bbvgg13_totalrepro",  "gobackseb_bbvgg16_fixeddo" ]



# for ix, bb in enumerate(backbones):
for exp_name in expnames:

        exp_group_name = "gobackseb_sweep"
        # exp_name = "{}_bb{}_do{}_avgpoolTrue_ms{}".format(model, bb, do, ms)
        # exp_name = "gobackseb_bb{}_totalrepro".format(bb)
        command_line_args = ""

        if mode == "test":
            command_line_args += "--config-path=$EXPERIMENT_ROOT_DIR/{} ".format(os.path.join(exp_group_name, exp_name, "hydra"))
            command_line_args += "exp.mode=test "
            command_line_args += "test.iid_set_split=all "

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
            command_line_args += "eval.confidence_measures.test=\"{}\" ".format(
                ["det_mcp", "det_pe", "mcd_mcp", "mcd_pe", "mcd_ee", "mcd_mi", "mcd_sv", "mcd_waic"])
        else:

            # if "devries" in model:
            #     command_line_args += "study={} ".format("cifar_devries_study")
            #     command_line_args += "eval.ext_confid_name={} ".format("devries")
            # else:
            command_line_args += "study={} ".format("cifar_tcp_confid_sweep")
            # command_line_args += "eval.ext_confid_name={} ".format("tcp")

            command_line_args += "data={} ".format("cifar10_data")
            command_line_args += "exp.group_name={} ".format(exp_group_name)
            command_line_args += "exp.name={} ".format(exp_name)
            command_line_args += "exp.mode={} ".format("train_test")
            # command_line_args += "model.dropout_rate={} ".format(do)
            # command_line_args += "model.name={} ".format(model)
            # command_line_args += "model.network.backbone={} ".format(bb)

            command_line_args += "model.network.imagenet_weights_path=null "
            # if cutout == False:
            #     command_line_args += "~data.augmentations.train.cutout "
            # command_line_args += "trainer.lr_scheduler.max_epochs={} ".format(sm)

            # command_line_args += "model.avg_pool={} ".format(avg_pool)

            # if ms == False:
            #     command_line_args += "trainer.callbacks.model_checkpoint={} ".format("null")
            #     command_line_args += "test.selection_criterion=latest "

            # if do:


            # else:
            #     command_line_args += "eval.confidence_measures.test=\"{}\" ".format(
            #         ["det_mcp", "det_pe", "ext"])

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