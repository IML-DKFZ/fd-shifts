from __future__ import annotations

import json
from itertools import islice
from typing import TYPE_CHECKING

import hydra
import numpy as np
import pl_bolts
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
from pytorch_lightning.utilities.parsing import AttributeDict
from rich import get_console
from rich.progress import track
from tqdm import tqdm

from fd_shifts import logger
from fd_shifts.utils import to_dict

if TYPE_CHECKING:
    from fd_shifts import configs


class net(pl.LightningModule):
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

        self.mean = torch.zeros((self.config.data.num_classes, self.model.num_features))
        self.icov = torch.eye(self.model.num_features)

        self.ext_confid_name = self.config.eval.ext_confid_name
        self.latent = []
        self.labels = []

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
            maha = None
            if any("ext" in cfd for cfd in self.query_confids.test):
                zm = z[:, None, :] - self.mean

                maha = -(torch.einsum("inj,jk,ink->in", zm, self.icov, zm))
                maha = maha.max(dim=1)[0].type_as(x).unsqueeze(1)
                conf_list.append(maha)

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
        probs = self.model.head(z)
        loss = torch.nn.functional.cross_entropy(probs, y)

        self.latent.append(z.cpu())
        self.labels.append(y.cpu())

        return {"loss": loss, "softmax": torch.softmax(probs, dim=1), "labels": y}

    def training_step_end(self, batch_parts):
        batch_parts["loss"] = batch_parts["loss"].mean()
        return batch_parts

    def training_epoch_end(self, outputs):
        with torch.no_grad():
            z = torch.cat(self.latent, dim=0)
            y = torch.cat(self.labels, dim=0)

            mean = []
            for c in y.unique():
                mean.append(z[y == c].mean(dim=0))

            mean = torch.stack(mean, dim=0)
            self.mean = mean
            self.icov = torch.inverse(
                torch.tensor(np.cov(z.numpy(), rowvar=False)).type_as(
                    self.model.head.weight
                )
            ).cpu()

        self.latent = []
        self.labels = []

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        if dataloader_idx > 0:
            y = y.fill_(0)
            y = y.long()

        z = self.model.forward_features(x)
        zm = z[:, None, :].cpu() - self.mean

        maha = -(torch.einsum("inj,jk,ink->in", zm, self.icov, zm))
        maha = maha.max(dim=1)[0]

        probs = self.model.head(z)
        loss = torch.nn.functional.cross_entropy(probs, y)

        return {
            "loss": loss,
            "softmax": torch.softmax(probs, dim=1),
            "labels": y,
            "confid": maha.type_as(x),
        }

    def validation_step_end(self, batch_parts):
        return batch_parts

    def on_test_start(self, *args):
        if not any("ext" in cfd for cfd in self.query_confids.test):
            return
        logger.info("Calculating trainset mean and cov")
        all_z = []
        all_y = []
        get_console().clear_live()
        tracker = track(
            self.trainer.datamodule.train_dataloader(), console=get_console()
        )

        if self.config.trainer.fast_dev_run:
            tracker = track(
                islice(
                    self.trainer.datamodule.train_dataloader(),
                    self.config.trainer.fast_dev_run,
                ),
                console=get_console(),
            )

        for x, y in tracker:
            x = x.type_as(self.model.head.weight)
            y = y.type_as(self.model.head.weight)
            z = self.model.forward_features(x)
            all_z.append(z.cpu())
            all_y.append(y.cpu())

        all_z = torch.cat(all_z, dim=0)
        all_y = torch.cat(all_y, dim=0)

        if torch.isnan(all_z).any():
            logger.error("NaN in z's: {}%", torch.isnan(all_z).any(dim=1).mean() * 100)

        mean = []
        for c in all_y.unique():
            mean.append(all_z[all_y == c].mean(dim=0))

        mean = torch.stack(mean, dim=0)
        self.mean = mean.type_as(self.model.head.weight)
        self.icov = torch.inverse(torch.cov(all_z.type_as(self.model.head.weight).T))

    def test_step(self, batch, batch_idx, *args):
        x, y = batch
        z = self.model.forward_features(x)

        maha = None
        if any("ext" in cfd for cfd in self.query_confids.test):
            zm = z[:, None, :] - self.mean

            maha = -(torch.einsum("inj,jk,ink->in", zm, self.icov, zm))
            maha = maha.max(dim=1)[0].type_as(x)
            # maha final ist abstand zu most likely class
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
            "confid": maha,
            "logits_dist": logits_dist,
            "confid_dist": confid_dist,
            "encoded": z,
        }

    def configure_optimizers(self):
        optim = hydra.utils.instantiate(
            self.config.trainer.optimizer,
            _convert_="all",
            _partial_=False,
            params=self.model.parameters(),
        )

        lr_sched = {
            "scheduler": hydra.utils.instantiate(self.config.trainer.lr_scheduler)(
                optimizer=optim
            ),
            "interval": "step",
        }

        optimizers = {
            "optimizer": optim,
            "lr_scheduler": lr_sched,
            "frequency": 1,
        }

        return optimizers

    def load_only_state_dict(self, path):
        ckpt = torch.load(path)
        logger.info("loading checkpoint from epoch {}".format(ckpt["epoch"]))
        self.load_state_dict(ckpt["state_dict"], strict=True)
