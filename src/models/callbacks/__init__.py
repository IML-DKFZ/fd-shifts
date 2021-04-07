from src.models.callbacks import confid_monitor
from src.models.callbacks import training_stages
from pytorch_lightning.callbacks import ModelCheckpoint

def get_callbacks(cf):
    """
        Return all queried callbacks
    """

    out_cb_list = []
    for k, v in cf.trainer.callbacks.items():
        if k == "model_checkpoint":
            for n_mc in range(v.n):
                out_cb_list.append(ModelCheckpoint(dirpath=cf.exp.version_dir,
                                          filename=v.filename[n_mc],
                                          monitor=v.selection_metric[n_mc],
                                          mode=v.mode[n_mc],
                                          save_top_k=v.save_top_k[n_mc],
                                          save_last=True,
                                         ))
        if k == "confid_monitor":
            out_cb_list.append(confid_monitor.ConfidMonitor(cf)) # todo explciit arguments!!!

        if k == "training_stages":
            out_cb_list.append(training_stages.TrainingStages(milestones=cf.trainer.callbacks.training_stages.milestones))

    return out_cb_list


