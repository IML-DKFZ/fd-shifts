import os
import subprocess
from itertools import product
import time

system_name = os.environ['SYSTEM_NAME']

current_dir = os.path.dirname(os.path.realpath(__file__))
exec_dir = "/".join(current_dir.split("/")[:-1])
exec_path = os.path.join(exec_dir,"exec.py")



train_mode = "train" # "test" / "train" / "analysis"
backbones = ["resnet50"] #
dropouts = [0] #[1, 0] # #
modes = ["confidnet", "devries", "dg"]
runs = [1]#[1 , 2 , 3, 4, 5]
lrs = [1e-2, 1e-3]#[1 , 2 , 3, 4, 5]
rewards = [2.2]#[2.2, 3, 6]
my_ix = 0

exp_name_list = []

for ix, (mode, bb, do, run, rew, lr) in enumerate(product(modes, backbones, dropouts, runs ,rewards, lrs)):

    if  not (mode=="devries" and do==1) and not (mode!="dg" and rew > 2.2):


        exp_group_name = "animals_check_sweep" # TODO CAREFUL CHANGED SWEEP NAME FOR CHECK
        exp_name = "{}_bb{}_do{}_run{}_rew{}_lr{}".format(mode, bb, do, run, rew, lr)
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
                    command_line_args += "study={} ".format("cifar_devries_study")
                    command_line_args += "model.network.name={} ".format("devries_and_enc")
                    command_line_args += "model.network.backbone={} ".format(bb)
                    command_line_args += "trainer.num_epochs={} ".format(18)
                    command_line_args += "trainer.optimizer.learning_rate={} ".format(lr)
                    command_line_args += "trainer.optimizer.weight_decay={} ".format(0.0001)


                elif mode == "dg":
                    command_line_args += "study={} ".format("dg_cifar_study")
                    command_line_args += "model.network.name={} ".format(bb)
                    command_line_args += "model.dg_reward={} ".format(rew)
                    command_line_args += "trainer.num_epochs={} ".format(18)
                    command_line_args += "trainer.dg_pretrain_epochs={} ".format(6)
                    command_line_args += "trainer.optimizer.learning_rate={} ".format(lr)
                    command_line_args += "trainer.optimizer.weight_decay={} ".format(0.0001)


                elif mode == "confidnet":
                    command_line_args += "study={} ".format("cifar_tcp_confid_sweep")
                    command_line_args += "model.network.name={} ".format("confidnet_and_enc")
                    command_line_args += "model.network.backbone={} ".format(bb)
                    command_line_args += "trainer.num_epochs={} ".format(25)
                    command_line_args += "trainer.num_epochs_backbone={} ".format(12)
                    command_line_args += "trainer.callbacks.training_stages.milestones=\"{}\" ".format([12, 22])
                    command_line_args += "trainer.learning_rate={} ".format(lr)
                    command_line_args += "trainer.learning_rate_confidnet={} ".format(1e-4)
                    command_line_args += "trainer.learning_rate_confidnet_finetune={} ".format(1e-7)
                    command_line_args += "trainer.weight_decay={} ".format(0.0001)


                command_line_args += "data={} ".format("wilds_animals_data")
                command_line_args += "exp.group_name={} ".format(exp_group_name)
                command_line_args += "exp.name={} ".format(exp_name)
                command_line_args += "exp.mode={} ".format("train_test")
                command_line_args += "model.dropout_rate={} ".format(do)
                command_line_args += "exp.global_seed={} ".format(run)
                command_line_args += "trainer.batch_size={} ".format(48)
                command_line_args += "model.network.imagenet_weights_path={} ".format("$EXPERIMENT_ROOT_DIR/pretrained_weights/resnet50-19c8e357.pth")
                command_line_args += "trainer.do_val=True "

                if bb == "resnet50":
                    command_line_args += "model.fc_dim={} ".format(2048)

                if do == 1:
                    command_line_args += "eval.confidence_measures.test=\"{}\" ".format(
                        ["det_mcp" , "det_pe", "ext", "ext_mcd", "ext_waic", "mcd_mcp", "mcd_pe", "mcd_ee", "mcd_mi", "mcd_sv", "mcd_waic"])
                else:
                    command_line_args += "eval.confidence_measures.train=\"{}\" ".format( #todo changed from paper sweep
                        ["det_mcp", "det_pe", "ext"])
                    command_line_args += "eval.confidence_measures.val=\"{}\" ".format( #todo changed from paper sweep
                        ["det_mcp", "det_pe", "ext"])
                    command_line_args += "eval.confidence_measures.test=\"{}\" ".format(
                        ["det_mcp", "det_pe", "ext"])

                command_line_args += "eval.query_studies.iid_study=wilds_animals "
                command_line_args += "~eval.query_studies.noise_study "
                command_line_args += "~eval.query_studies.new_class_study "
                command_line_args += "+eval.query_studies.in_class_study=\"{}\" ".format(['wilds_animals_ood_test'])


            if system_name == "cluster":

                launch_command = ""
                launch_command += "bsub "
                launch_command += "-gpu num=1:"
                launch_command += "j_exclusive=yes:"
                launch_command += "mode=exclusive_process:"
                # launch_command += "gmodel=TITANXp:"
                launch_command += "gmem=23G "
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
            time.sleep(2)

print(my_ix)
print(exp_name_list)
#subprocess.call("python {}/utils/job_surveillance.py --in_name {} --out_name {} --n_jobs {} &".format(exec_dir, log_folder, sur_out_name, job_ix), shell=True)
