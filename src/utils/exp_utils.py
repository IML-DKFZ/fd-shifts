import os
import torch
import random
import numpy as np
import sys
import pytorch_lightning as pl

def set_seed(seed):
    print("SETTING GLOBAL SEED")
    pl.seed_everything(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_next_version(exp_dir):
    # get best.ckpt of experiment. if split over multiple runs (e.g. due to resuming), still find the best.ckpt.
    # if there are multiple overall runs in the folder select the latest.
    ver_list = [int(x.split("_")[1]) for x in os.listdir(exp_dir) if "version_" in x]
    if len(ver_list) == 0:
        return 0
    max_ver = max(ver_list)
    return max_ver + 1

def get_most_recent_version(exp_dir):
    # get best.ckpt of experiment. if split over multiple runs (e.g. due to resuming), still find the best.ckpt.
    # if there are multiple overall runs in the folder select the latest.
    ver_list = [int(x.split("_")[1]) for x in os.listdir(exp_dir) if "version_" in x]
    if len(ver_list) == 0:
        RuntimeError("No checkpoints exist in this experiment dir!")
    max_ver = max(ver_list)
    return max_ver

def get_ckpt_path_from_previous_version(exp_dir,version, selection_criterion):
    if selection_criterion == "latest":
        selection_criterion = "last"
    resume_ckpt = os.path.join(exp_dir, "version_{}".format(version), "{}.ckpt".format(selection_criterion))
    if not os.path.isfile(resume_ckpt):
        RuntimeError("requested resume ckpt does not exist.")
    return resume_ckpt


def get_path_to_best_ckpt(exp_dir, selection_criterion, selection_mode):
    path_list = []
    for r,d,f in os.walk(exp_dir):
        path_list.extend([os.path.join(r, x) for x in f if selection_criterion in x])

    if len(path_list) == 1:
        return path_list[0]
    else:
        scores_list = [list(torch.load(p)["callbacks"].values())[0]["best_model_score"].item() for p in path_list]
        if selection_mode == "min":
            return path_list[scores_list.index(min(scores_list))]
        else:
            return path_list[scores_list.index(max(scores_list))]




class Logger(object):
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


