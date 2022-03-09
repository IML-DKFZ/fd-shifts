import os
import subprocess
from itertools import product
import time


current_dir = os.path.dirname(os.path.realpath(__file__))
exec_dir = "/".join(current_dir.split("/")[:-1])
exec_path = os.path.join(exec_dir,"exec.py")



train_mode = "train" # "test" / "train" / "analysis"
backbones = ["resnet50"] #
dropouts = [0, 1]
modes = ["dg", "confidnet", "devries"]
runs = [1 , 2 , 3, 4, 5]
rewards = [2.2, 3, 6, 10, 15]
my_ix = 0

exp_name_list = []

for ix, (mode, bb, do, run, rew) in enumerate(product(modes, backbones, dropouts, runs ,rewards)):

    if  not (mode!="dg" and rew > 2.2):


        exp_group_name = "animals_paper_sweep"
        exp_name = "{}_bb{}_do{}_run{}_rew{}".format(mode, bb, do, run, rew)
        exp_name_list.append(exp_name)
        if 1==1:
            my_ix += 1
            command_line_args = ""

            if train_mode == "test":
                command_line_args += "--config-path=$EXPERIMENT_ROOT_DIR/{} ".format(os.path.join(exp_group_name, exp_name, "hydra"))
                command_line_args += "exp.mode=test "


            elif train_mode == "analysis":
                command_line_args += "--config-path=$EXPERIMENT_ROOT_DIR/{} ".format(os.path.join(exp_group_name, exp_name, "hydra"))
                command_line_args += "exp.mode=analysis "

            else:


                if mode == "devries":
                    command_line_args += "study={} ".format("devries")
                    command_line_args += "model.network.name={} ".format("devries_and_enc")
                    command_line_args += "model.network.backbone={} ".format(bb)
                    command_line_args += "trainer.num_epochs={} ".format(12)
                    command_line_args += "trainer.optimizer.learning_rate={} ".format(0.001)
                    command_line_args += "trainer.optimizer.weight_decay={} ".format(0)


                elif mode == "dg":
                    command_line_args += "study={} ".format("deepgamblers")
                    command_line_args += "model.network.name={} ".format(bb)
                    command_line_args += "model.dg_reward={} ".format(rew)
                    command_line_args += "trainer.num_epochs={} ".format(18)
                    command_line_args += "trainer.dg_pretrain_epochs={} ".format(6)
                    command_line_args += "trainer.optimizer.learning_rate={} ".format(0.001)
                    command_line_args += "trainer.optimizer.weight_decay={} ".format(0)


                elif mode == "confidnet":
                    command_line_args += "study={} ".format("confidnet")
                    command_line_args += "model.network.name={} ".format("confidnet_and_enc")
                    command_line_args += "model.network.backbone={} ".format(bb)
                    command_line_args += "trainer.num_epochs={} ".format(20)
                    command_line_args += "trainer.num_epochs_backbone={} ".format(12)
                    command_line_args += "trainer.callbacks.training_stages.milestones=\"{}\" ".format([12, 17])
                    command_line_args += "trainer.learning_rate={} ".format(0.001)
                    command_line_args += "trainer.weight_decay={} ".format(0)


                command_line_args += "data={} ".format("wilds_animals_data")
                command_line_args += "exp.group_name={} ".format(exp_group_name)
                command_line_args += "exp.name={} ".format(exp_name)
                command_line_args += "exp.mode={} ".format("train_test")
                command_line_args += "model.dropout_rate={} ".format(do)
                command_line_args += "exp.global_seed={} ".format(run)
                command_line_args += "trainer.batch_size={} ".format(16)
                command_line_args += "model.network.imagenet_weights_path={} ".format("$EXPERIMENT_ROOT_DIR/pretrained_weights/resnet50-19c8e357.pth")
                command_line_args += "trainer.do_val=True "

                if bb == "resnet50":
                    command_line_args += "model.fc_dim={} ".format(2048)

                if do == 1:
                    command_line_args += "eval.confidence_measures.test=\"{}\" ".format(
                        ["det_mcp" , "det_pe", "ext", "ext_mcd", "ext_waic", "mcd_mcp", "mcd_pe", "mcd_ee", "mcd_mi", "mcd_sv", "mcd_waic"])
                else:
                    command_line_args += "eval.confidence_measures.train=\"{}\" ".format(
                        ["det_mcp", "det_pe", "ext"])
                    command_line_args += "eval.confidence_measures.val=\"{}\" ".format(
                        ["det_mcp", "det_pe", "ext"])
                    command_line_args += "eval.confidence_measures.test=\"{}\" ".format(
                        ["det_mcp", "det_pe", "ext"])

                command_line_args += "eval.query_studies.iid_study=wilds_animals "
                command_line_args += "~eval.query_studies.noise_study "
                command_line_args += "~eval.query_studies.new_class_study "
                command_line_args += "+eval.query_studies.in_class_study=\"{}\" ".format(['wilds_animals_ood_test'])


            launch_command = "python -u {} ".format(exec_path)
            launch_command += command_line_args
            print("Launch command: ", launch_command)
            subprocess.call(launch_command, shell=True)
            time.sleep(2)

print(my_ix)
print(exp_name_list)
