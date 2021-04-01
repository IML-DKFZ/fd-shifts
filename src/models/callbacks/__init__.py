from src.models.callbacks import confid_monitor
from pytorch_lightning.callbacks import ModelCheckpoint

def get_callbacks(cf):
    """
        Return all queried callbacks
    """

    # Available models
    callback_factory = {
        "confid_monitor": confid_monitor.ConfidMonitor(cf), # todo explciit arguments!!!
        "model_checkpoint": ModelCheckpoint(dirpath=cf.exp.version_dir,
                                          filename="best",
                                          monitor=cf.trainer.selection_metric,
                                          mode=cf.trainer.selection_mode,
                                          save_top_k=cf.trainer.save_top_k,
                                          save_last=True,
                                         )
    }

    out_cb_list = []
    for cb in cf.model.callbacks:
        out_cb_list.append(callback_factory[cb])

    return out_cb_list


