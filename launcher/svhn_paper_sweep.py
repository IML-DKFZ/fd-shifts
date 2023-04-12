import os
import subprocess
import time
from itertools import product

current_dir = os.path.dirname(os.path.realpath(__file__))
exec_dir = "/".join(current_dir.split("/")[:-1])
exec_path = os.path.join(exec_dir, "exec.py")


train_mode = "train"  # "test" / "train" / "analysis"
backbones = ["svhn_small_conv"]  #
dropouts = [1, 0]  # #
modes = ["confidnet", "dg", "devries"]
runs = [1, 2, 3, 4, 5]
rewards = [2.2, 3, 6, 10]
my_ix = 0


exp_name_list = []

for ix, (mode, bb, do, run, rew) in enumerate(
    product(modes, backbones, dropouts, runs, rewards)
):
    if not (mode != "dg" and rew > 2.2):
        exp_group_name = "svhn_paper_sweep"
        exp_name = "{}_bb{}_do{}_run{}_rew{}".format(mode, bb, do, run, rew)
        exp_name_list.append(exp_name)
        if 1 == 1:
            my_ix += 1
            command_line_args = ""

            if train_mode == "test":
                command_line_args += "--config-path=$EXPERIMENT_ROOT_DIR/{} ".format(
                    os.path.join(exp_group_name, exp_name, "hydra")
                )
                command_line_args += "exp.mode=test "

            elif train_mode == "analysis":
                command_line_args += "--config-path=$EXPERIMENT_ROOT_DIR/{} ".format(
                    os.path.join(exp_group_name, exp_name, "hydra")
                )
                command_line_args += "exp.mode=analysis "

            else:
                if mode == "devries":
                    command_line_args += "study={} ".format("devries")
                    command_line_args += "model.network.name={} ".format(
                        "devries_and_enc"
                    )
                    command_line_args += "model.network.backbone={} ".format(bb)
                    command_line_args += "trainer.num_epochs={} ".format(100)
                    command_line_args += "trainer.optimizer.learning_rate={} ".format(
                        0.01
                    )

                elif mode == "dg":
                    command_line_args += "study={} ".format("deepgamblers")
                    command_line_args += "model.network.name={} ".format(bb)
                    command_line_args += "model.dg_reward={} ".format(rew)
                    command_line_args += "trainer.num_epochs={} ".format(150)
                    command_line_args += "trainer.dg_pretrain_epochs={} ".format(50)
                    command_line_args += "trainer.optimizer.learning_rate={} ".format(
                        0.01
                    )

                elif mode == "confidnet":
                    command_line_args += "study={} ".format("confidnet")
                    command_line_args += "model.network.name={} ".format(
                        "confidnet_and_enc"
                    )
                    command_line_args += "model.network.backbone={} ".format(bb)
                    command_line_args += "trainer.num_epochs={} ".format(320)
                    command_line_args += "trainer.num_epochs_backbone={} ".format(100)
                    command_line_args += (
                        'trainer.callbacks.training_stages.milestones="{}" '.format(
                            [100, 300]
                        )
                    )
                    command_line_args += "trainer.learning_rate={} ".format(0.01)

                command_line_args += "data={} ".format("svhn_data")
                command_line_args += "exp.group_name={} ".format(exp_group_name)
                command_line_args += "exp.name={} ".format(exp_name)
                command_line_args += "exp.mode={} ".format("train_test")
                command_line_args += "model.dropout_rate={} ".format(do)
                command_line_args += "exp.global_seed={} ".format(run)

                if do == 1:
                    command_line_args += 'eval.confidence_measures.test="{}" '.format(
                        [
                            "det_mcp",
                            "det_pe",
                            "ext",
                            "ext_mcd",
                            "ext_waic",
                            "mcd_mcp",
                            "mcd_pe",
                            "mcd_ee",
                            "mcd_mi",
                            "mcd_sv",
                            "mcd_waic",
                        ]
                    )
                else:
                    command_line_args += 'eval.confidence_measures.test="{}" '.format(
                        ["det_mcp", "det_pe", "ext"]
                    )

                command_line_args += "eval.query_studies.iid_study=svhn "
                command_line_args += "~eval.query_studies.noise_study "
                command_line_args += 'eval.query_studies.new_class_study="{}" '.format(
                    ["tinyimagenet_resize", "cifar10", "cifar100"]
                )

            launch_command = "python -u {} ".format(exec_path)
            launch_command += command_line_args
            print("Launch command: ", launch_command)
            subprocess.call(launch_command, shell=True)
            time.sleep(2)

print(my_ix)
print(exp_name_list)
