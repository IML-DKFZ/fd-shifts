from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from fd_shifts import logger
from fd_shifts.models.networks import get_network
from fd_shifts.models.networks.resnet50_imagenet import ResNetEncoder
from fd_shifts.utils import to_dict

if TYPE_CHECKING:
    from fd_shifts import configs


class Module(pl.LightningModule):
    """

    Attributes:
        conf:
        test_mcd_samples:
        monitor_mcd_samples:
        learning_rate:
        learning_rate_confidnet:
        learning_rate_confidnet_finetune:
        lr_scheduler:
        momentum:
        weight_decay:
        query_confids:
        num_epochs:
        pretrained_backbone_path:
        pretrained_confidnet_path:
        confidnet_lr_scheduler:
        imagenet_weights_path:
        loss_ce:
        loss_mse:
        ext_confid_name:
        network:
        backbone:
        training_stage:
        test_results:
        loaded_epoch:
    """

    def __init__(self, cf: configs.Config):
        super().__init__()

        self.save_hyperparameters(to_dict(cf))
        self.conf = cf

        self.test_mcd_samples = cf.model.test_mcd_samples
        self.monitor_mcd_samples = cf.model.monitor_mcd_samples
        self.lr_scheduler = cf.trainer.lr_scheduler
        self.query_confids = cf.eval.confidence_measures
        self.num_epochs = cf.trainer.num_epochs
        if cf.trainer.callbacks["model_checkpoint"] is not None:
            logger.info(
                "Initializing custom Model Selector. {}",
                cf.trainer.callbacks["model_checkpoint"],
            )
            self.selection_metrics = cf.trainer.callbacks["model_checkpoint"][
                "selection_metric"
            ]
            self.selection_modes = cf.trainer.callbacks["model_checkpoint"]["mode"]
            self.test_selection_criterion = cf.test.selection_criterion
        self.pretrained_backbone_path = cf.trainer.callbacks["training_stages"][
            "pretrained_backbone_path"
        ]
        self.pretrained_confidnet_path = cf.trainer.callbacks["training_stages"][
            "pretrained_confidnet_path"
        ]
        self.confidnet_lr_scheduler = cf.trainer.callbacks["training_stages"][
            "confidnet_lr_scheduler"
        ]
        self.imagenet_weights_path = cf.model.network.imagenet_weights_path

        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_mse = nn.MSELoss(reduction="sum")
        self.ext_confid_name = cf.eval.ext_confid_name

        self.network = get_network(cf.model.network.name)(cf)

        assert (backbone_name := cf.model.network.backbone)
        self.backbone = get_network(backbone_name)(cf)

        self.training_stage = 0

        self.test_results: dict[str, torch.Tensor | None] = {}

    # pylint: disable=arguments-differ
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): batch of images

        Returns:
            Predicted probabilities
        """
        return self.network(x)

    def mcd_eval_forward(
        self, x: torch.Tensor, n_samples: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): batch of images
            n_samples (int): number of montecarlo dropout samples to draw

        Returns:
            batch of samples of predicted probabilities
        """
        self.network.encoder.enable_dropout()
        self.backbone.encoder.enable_dropout()

        softmax_list: list[torch.Tensor] = []
        conf_list: list[torch.Tensor] = []
        for _ in range(n_samples - len(softmax_list)):
            logits = self.backbone(x)
            _, confidence = self.network(x)
            confidence = torch.sigmoid(confidence).squeeze(1)
            softmax_list.append(logits.unsqueeze(2))
            conf_list.append(confidence.unsqueeze(1))

        self.network.encoder.disable_dropout()
        self.backbone.encoder.disable_dropout()

        return torch.cat(softmax_list, dim=2), torch.cat(conf_list, dim=1)

    def on_train_start(self) -> None:
        if self.imagenet_weights_path and isinstance(
            self.backbone.encoder, ResNetEncoder
        ):
            self.backbone.encoder.load_pretrained_imagenet_params(
                self.imagenet_weights_path
            )

        for ix, x in enumerate(self.backbone.named_modules()):
            tqdm.write(str(ix))
            tqdm.write(str(x[1]))

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor | None]:
        """
        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): one batch of images and labels
            batch_idx (int): index of current batch

        Returns:
            dict containing loss, softmax, labels and confidences

        Raises:
            ValueError: if somehow the training stage goes beyond 2
        """
        if self.training_stage == 0:
            x, y = batch
            logits = self.backbone(x)
            loss = self.loss_ce(logits, y)
            softmax = F.softmax(logits, dim=1)
            return {"loss": loss, "softmax": softmax, "labels": y, "confid": None}

        if self.training_stage == 1:
            x, y = batch
            outputs = self.network(x)
            softmax = F.softmax(outputs[0], dim=1)
            pred_confid = torch.sigmoid(outputs[1])
            tcp = softmax.gather(1, y.unsqueeze(1))
            loss = F.mse_loss(pred_confid, tcp)
            return {
                "loss": loss,
                "softmax": softmax,
                "labels": y,
                "confid": pred_confid.squeeze(1),
            }

        if self.training_stage == 2:
            x, y = batch
            softmax = F.softmax(self.backbone(x), dim=1)
            _, pred_confid = self.network(x)
            pred_confid = torch.sigmoid(pred_confid)
            tcp = softmax.gather(1, y.unsqueeze(1))
            loss = F.mse_loss(pred_confid, tcp)
            return {
                "loss": loss,
                "softmax": softmax,
                "labels": y,
                "confid": pred_confid.squeeze(1),
            }

        raise ValueError("There is no training stage larger than 2")

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor | None]:
        """
        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): one batch of images and labels
            batch_idx (int): index of current batch

        Returns:
            dict containing loss, softmax, labels and confidences

        Raises:
            ValueError: if somehow the training stage goes beyond 2
        """

        if self.training_stage == 0:
            x, y = batch
            logits = self.backbone(x)
            loss = self.loss_ce(logits, y)
            softmax = F.softmax(logits, dim=1)
            return {"loss": loss, "softmax": softmax, "labels": y, "confid": None}

        if self.training_stage == 1:
            x, y = batch
            outputs = self.network(x)
            softmax = F.softmax(outputs[0], dim=1)
            pred_confid = torch.sigmoid(outputs[1])
            tcp = softmax.gather(1, y.unsqueeze(1))
            loss = F.mse_loss(pred_confid, tcp)
            return {
                "loss": loss,
                "softmax": softmax,
                "labels": y,
                "confid": pred_confid.squeeze(1),
            }

        if self.training_stage == 2:
            x, y = batch
            softmax = F.softmax(self.backbone(x), dim=1)
            _, pred_confid = self.network(x)
            pred_confid = torch.sigmoid(pred_confid)
            tcp = softmax.gather(1, y.unsqueeze(1))
            loss = F.mse_loss(pred_confid, tcp)

            softmax_dist = None
            return {
                "loss": loss,
                "softmax": softmax,
                "softmax_dist": softmax_dist,
                "labels": y,
                "confid": pred_confid.squeeze(1),
            }

        raise ValueError("There is no training stage larger than 2")

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_id: int | None = None,
    ) -> dict[str, torch.Tensor | None]:
        x, y = batch

        z = self.backbone.forward_features(x)
        logits = self.backbone.head(z)
        _, pred_confid = self.network(x)
        pred_confid = torch.sigmoid(pred_confid).squeeze(1)

        logits_dist = None
        pred_confid_dist = None

        if any("mcd" in cfd for cfd in self.query_confids.test) and (
            not (self.conf.test.compute_train_encodings and dataloader_id == 0)
        ):
            logits_dist, pred_confid_dist = self.mcd_eval_forward(
                x=x, n_samples=self.test_mcd_samples
            )

        return {
            "logits": logits,
            "logits_dist": logits_dist,
            "labels": y,
            "confid": pred_confid,
            "confid_dist": pred_confid_dist,
            "encoded": z,
        }

    def configure_optimizers(
        self,
    ) -> tuple[
        list[torch.optim.Optimizer], list[torch.optim.lr_scheduler._LRScheduler]
    ]:
        optimizers = [
            hydra.utils.instantiate(self.conf.trainer.optimizer, _partial_=True)(
                self.backbone.parameters()
            )
        ]

        schedulers = [
            hydra.utils.instantiate(self.conf.trainer.lr_scheduler)(
                optimizer=optimizers[0]
            )
        ]

        return optimizers, schedulers

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.loaded_epoch = checkpoint["epoch"]
        logger.info("loading checkpoint at epoch {}".format(self.loaded_epoch))

    def load_only_state_dict(self, path: str | Path) -> None:
        ckpt = torch.load(path)

        pattern = re.compile(r"^(\w*\.)(encoder|classifier)(\..*)")

        # For backwards-compatibility with before commit 1bdc717
        for param in list(ckpt["state_dict"].keys()):
            if param.startswith(
                "backbone.classifier.module.model.features"
            ) or param.startswith("network.classifier.module.model.features"):
                del ckpt["state_dict"][param]
                continue
            if param.startswith(
                "backbone.classifier.module.model.classifier"
            ) or param.startswith("network.classifier.module.model.classifier"):
                correct_param = param.replace(".model.classifier", "")
                ckpt["state_dict"][correct_param] = ckpt["state_dict"][param]
                del ckpt["state_dict"][param]
                param = correct_param
            if pattern.match(param):
                correct_param = re.sub(pattern, r"\1_\2\3", param)
                ckpt["state_dict"][correct_param] = ckpt["state_dict"][param]
                del ckpt["state_dict"][param]

        logger.info("loading checkpoint from epoch {}".format(ckpt["epoch"]))
        self.load_state_dict(ckpt["state_dict"], strict=False)

    def last_layer(self):
        state = self.state_dict()
        model_prefix = "backbone"
        if f"{model_prefix}._classifier.module.weight" in state:
            w = state[f"{model_prefix}._classifier.module.weight"]
            b = state[f"{model_prefix}._classifier.module.bias"]
        elif f"{model_prefix}._classifier.fc.weight" in state:
            w = state[f"{model_prefix}._classifier.fc.weight"]
            b = state[f"{model_prefix}._classifier.fc.bias"]
        elif f"{model_prefix}._classifier.fc2.weight" in state:
            w = state[f"{model_prefix}._classifier.fc2.weight"]
            b = state[f"{model_prefix}._classifier.fc2.bias"]
        else:
            print(list(state.keys()))
            raise RuntimeError("No classifier weights found")

        return w, b
