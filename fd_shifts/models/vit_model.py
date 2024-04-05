from __future__ import annotations

from typing import TYPE_CHECKING

import lightning as L
import timm
import torch
import torch.nn as nn

from fd_shifts import logger
from fd_shifts.utils import to_dict

if TYPE_CHECKING:
    from fd_shifts import configs


class net(L.LightningModule):
    """Vision Transformer module"""

    def __init__(self, cfg: configs.Config):
        super().__init__()

        self.save_hyperparameters(to_dict(cfg))

        self.config = cfg

        self.model = timm.create_model(
            "vit_base_patch16_224_in21k",
            pretrained=True,
            img_size=self.config.data.img_size[0],
            num_classes=self.config.data.num_classes,
            drop_rate=self.config.model.dropout_rate * 0.1,
        )
        self.model.reset_classifier(self.config.data.num_classes)
        self.model.head.weight.tensor = torch.zeros_like(self.model.head.weight)
        self.model.head.bias.tensor = torch.zeros_like(self.model.head.bias)

        self.ext_confid_name = self.config.eval.ext_confid_name

        self.query_confids = cfg.eval.confidence_measures
        self.test_mcd_samples = 50

    def disable_dropout(self):
        for layer in self.named_modules():
            if isinstance(layer[1], nn.modules.dropout.Dropout):
                layer[1].eval()

    def enable_dropout(self):
        for layer in self.named_modules():
            if isinstance(layer[1], nn.modules.dropout.Dropout):
                layer[1].train()

    def mcd_eval_forward(self, x, n_samples):
        self.enable_dropout()

        softmax_list = []
        conf_list = []
        for _ in range(n_samples - len(softmax_list)):
            z = self.model.forward_features(x)
            probs = self.model.head(z)

            softmax_list.append(probs.unsqueeze(2))

        self.disable_dropout()

        return (
            torch.cat(softmax_list, dim=2),
            torch.cat(conf_list, dim=1) if len(conf_list) else None,
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.model.forward_features(x)
        probs = self.model.forward_head(z)
        loss = torch.nn.functional.cross_entropy(probs, y)

        return {"loss": loss, "softmax": torch.softmax(probs, dim=1), "labels": y}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        z = self.model.forward_features(x)
        probs = self.model.forward_head(z)
        loss = torch.nn.functional.cross_entropy(probs, y)

        return {
            "loss": loss,
            "softmax": torch.softmax(probs, dim=1),
            "labels": y,
            "confid": None,
        }

    def validation_step_end(self, batch_parts):
        return batch_parts

    def test_step(self, batch, batch_idx, *args):
        x, y = batch
        z = self.model.forward_features(x)
        z = self.model.forward_head(z, pre_logits=True)
        probs = self.model.head(z)

        logits_dist = None
        confid_dist = None
        if any("mcd" in cfd for cfd in self.query_confids.test):
            logits_dist, confid_dist = self.mcd_eval_forward(
                x=x, n_samples=self.test_mcd_samples
            )

        return {
            "logits": probs,
            "labels": y,
            "confid": None,
            "logits_dist": logits_dist,
            "confid_dist": confid_dist,
            "encoded": z,
        }

    def configure_optimizers(self):
        optim = self.config.trainer.optimizer(self.model.parameters())

        lr_sched = [
            {
                "scheduler": self.config.trainer.lr_scheduler(optim),
                "interval": "step",
            }
        ]

        optimizers = [optim]

        return optimizers, lr_sched

    def load_only_state_dict(self, path):
        ckpt = torch.load(path)
        logger.info("loading checkpoint from epoch {}".format(ckpt["epoch"]))
        self.load_state_dict(ckpt["state_dict"], strict=True)
