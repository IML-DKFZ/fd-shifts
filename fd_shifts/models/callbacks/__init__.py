from pytorch_lightning import Callback
from pytorch_lightning.callbacks import (
    GPUStatsMonitor,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)

from fd_shifts import configs, logger
from fd_shifts.models.callbacks import confid_monitor, training_stages


def get_callbacks(cfg: configs.Config) -> list[Callback]:
    """Dynamically get needed callbacks
    Args:
        cfg (configs.Config): config object

    Returns:
        all queried callbacks
    """

    out_cb_list = []
    for k, v in cfg.trainer.callbacks.items():
        if k == "model_checkpoint":
            if hasattr(v, "n"):
                for n_mc in range(v.n):
                    out_cb_list.append(
                        ModelCheckpoint(
                            dirpath=cfg.exp.version_dir,
                            filename=v.filename[n_mc],
                            monitor=v.selection_metric[n_mc],
                            mode=v.mode[n_mc],
                            save_top_k=v.save_top_k[n_mc],
                            save_last=True,
                            verbose=False,
                        )
                    )
            else:
                logger.info("Adding ModelCheckpoint callback")
                out_cb_list.append(
                    ModelCheckpoint(
                        dirpath=cfg.exp.version_dir,
                        save_last=True,
                        every_n_train_steps=cfg.trainer.num_steps // 2
                        if cfg.trainer.num_steps is not None
                        else None,
                    )
                )

        if k == "confid_monitor":
            out_cb_list.append(confid_monitor.ConfidMonitor(cfg))

        if k == "training_stages":
            out_cb_list.append(
                training_stages.TrainingStages(
                    milestones=v["milestones"],
                    disable_dropout_at_finetuning=v["disable_dropout_at_finetuning"],
                )
            )

        if k == "learning_rate_monitor":
            out_cb_list.append(LearningRateMonitor(logging_interval="epoch"))

    return out_cb_list
