import os
import subprocess
from itertools import product


def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    else:
        return False

#system_name = os.environ["SYSTEM_NAME"]

mode = "train_test"
delete_previous_checkpoints = True if "train" in mode else False  # has to be false for test mode!
check_pretrained_model = True if "train" in mode else False # has to be true for slurm submissions!
apply_test_mods = False if "train" in mode else True

exp_cluster_name = "cluster_test"
exp_group_name = "group_test"
exp_group_dir = os.path.join(exp_cluster_name, exp_group_name)
log_folder = os.path.join(exp_group_dir, "log")
sur_out_name = os.path.join(exp_group_dir, "surveillance_sheet.txt")

for f in [exp_group_dir, log_folder, sur_out_name]:
    if not os.path.exists(f):
        os.makedirs(f)

current_dir = os.path.dirname(os.path.realpath(__file__))
exec_dir = "/".join(current_dir.split("/")[:-2])
exec_path = os.path.join(exec_dir,"exec.py")
dataset_source = os.path.join(exec_dir, "experiments", "clevr")


job_ix = 0

launch_command = ""
launch_command += "bsub "
launch_command += "-gpu num=1:"
launch_command += "j_exclusive=yes:"
launch_command += "mode=exclusive_process:"
launch_command += "gmem=10.7G "
launch_command += "-L /bin/bash -q gpu "
launch_command += "'source ~/.bashrc && "
launch_command += "source ~/.virtualenvs/confid/bin/activate && "

#for gm, aenc, adec, mm, bns, bp, ns, pr, xd in product(graph_models, attr_encode, attr_decode, missing_attr_mode, bn_stats_mode, bias_pickle, num_steps, perceptual_reco, x_drop):
#        if job_ix < 1000000 and not (pr == True and xd == True) and not (mm == "poe" and aenc == "nested_khot"): # job_ix > ... not possible, because only is increased for running jobs
#job_name = "{}_{}_{}_{}_{}_{}_{}_{}_{}".format(gm, aenc, adec, mm, bns, bp, ns, pr, xd)
job_name = "test_job"

exp_dir = os.path.join(exp_group_dir, job_name)
if delete_previous_checkpoints:
    try:
        os.remove(os.path.join(exp_dir, "fold_0", "last_checkpoint.pth"))
        print("removed previous checkpoint.{}".format(job_name))
    except:
        pass

launch_command += "python -u {} --exec_dir {} --exp_dir {} --dataset_source {} --mode {} "\
    .format(exec_path, exec_dir, exp_dir, dataset_source, mode)

#launch_command += " --{} {} ".format("graphical_model", gm)

if mode == "test":
    launch_command += "--use_stored_setting"

if check_pretrained_model:
    launch_command += "--check_pretrained_model "

if apply_test_mods:
    launch_command += "--apply_test_config_mods "

subprocess.call(launch_command + "'", shell=True)

#subprocess.call("python {}/utils/job_surveillance.py --in_name {} --out_name {} --n_jobs {} &".format(exec_dir, log_folder, sur_out_name, job_ix), shell=True)