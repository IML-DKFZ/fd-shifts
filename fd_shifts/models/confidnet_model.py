from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import lightning as L
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


class Module(L.LightningModule):
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
        self.automatic_optimization = False

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

        self.milestones = cf.trainer.callbacks["training_stages"]["milestones"]
        self.disable_dropout_at_finetuning = cf.trainer.callbacks["training_stages"][
            "disable_dropout_at_finetuning"
        ]

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

        if self.pretrained_backbone_path is not None:
            self.milestones[1] = self.milestones[1] - self.milestones[0]
            self.milestones[0] = 0

    def on_train_epoch_start(self):
        if (
            self.current_epoch == self.milestones[0]
        ):  # this is the end before the queried epoch
            logger.info("Starting Training ConfidNet")
            self.training_stage = 1
            if (
                self.pretrained_backbone_path is None
            ):  # trained from scratch, reload best epoch
                best_ckpt_path = self.trainer.checkpoint_callbacks[
                    0
                ].last_model_path  # No backbone model selection!!
                logger.info("Check last backbone path {}", best_ckpt_path)
            else:
                best_ckpt_path = self.pretrained_backbone_path

            loaded_ckpt = torch.load(best_ckpt_path)
            loaded_state_dict = loaded_ckpt["state_dict"]

            backbone_encoder_state_dict = OrderedDict(
                (k.replace("backbone.encoder.", ""), v)
                for k, v in loaded_state_dict.items()
                if "backbone.encoder." in k
            )
            if len(backbone_encoder_state_dict) == 0:
                backbone_encoder_state_dict = loaded_state_dict
            backbone_classifier_state_dict = OrderedDict(
                (k.replace("backbone.classifier.", ""), v)
                for k, v in loaded_state_dict.items()
                if "backbone.classifier." in k
            )

            self.backbone.encoder.load_state_dict(
                backbone_encoder_state_dict, strict=True
            )
            self.backbone.classifier.load_state_dict(
                backbone_classifier_state_dict, strict=True
            )
            self.network.encoder.load_state_dict(
                backbone_encoder_state_dict, strict=True
            )
            self.network.classifier.load_state_dict(
                backbone_classifier_state_dict, strict=True
            )

            logger.info(
                "loaded checkpoint {} from epoch {} into backbone and network.".format(
                    best_ckpt_path, loaded_ckpt["epoch"]
                )
            )

            self.network.encoder = deepcopy(self.backbone.encoder)
            self.network.classifier = deepcopy(self.backbone.classifier)

            logger.info("freezing backbone and enabling confidnet")
            self.freeze_layers(self.backbone.encoder)
            self.freeze_layers(self.backbone.classifier)
            self.freeze_layers(self.network.encoder)
            self.freeze_layers(self.network.classifier)

        if self.current_epoch >= self.milestones[0]:
            self.disable_bn(self.backbone.encoder)
            self.disable_bn(self.network.encoder)
            for param_group in trainer.optimizers[0].param_groups:
                logger.info("CHECK ConfidNet RATE {}", param_group["lr"])

        if self.current_epoch == self.milestones[1]:
            logger.info(
                "Starting Training Fine Tuning ConfidNet"
            )  # new optimizer or add param groups? both adam according to paper!
            self.training_stage = 2
            if self.pretrained_confidnet_path is not None:
                best_ckpt_path = self.pretrained_confidnet_path
            elif (
                hasattr(self, "test_selection_criterion")
                and "latest" not in self.test_selection_criterion
            ):
                best_ckpt_path = trainer.checkpoint_callbacks[1].best_model_path
                logger.info(
                    "Test selection criterion {}", self.test_selection_criterion
                )
                logger.info("Check BEST confidnet path {}", best_ckpt_path)
            else:
                best_ckpt_path = None
                logger.info("going with latest confidnet")
            if best_ckpt_path is not None:
                loaded_ckpt = torch.load(best_ckpt_path)
                loaded_state_dict = loaded_ckpt["state_dict"]
                loaded_state_dict = OrderedDict(
                    (k.replace("network.confid_net.", ""), v)
                    for k, v in loaded_state_dict.items()
                    if "network.confid_net" in k
                )
                self.network.confid_net.load_state_dict(loaded_state_dict, strict=True)
                logger.info(
                    "loaded checkpoint {} from epoch {} into new encoder".format(
                        best_ckpt_path, loaded_ckpt["epoch"]
                    )
                )

            self.unfreeze_layers(self.network.encoder)

        if self.disable_dropout_at_finetuning:
            if self.current_epoch >= self.milestones[1]:
                self.disable_dropout(self.backbone.encoder)
                self.disable_dropout(self.network.encoder)

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
        optimizer = self.optimizers()[self.training_stage]

        if self.training_stage == 0:
            lr_sched = self.lr_schedulers()[0]

            x, y = batch
            logits = self.backbone(x)
            loss = self.loss_ce(logits, y) / self.conf.trainer.accumulate_grad_batches
            self.manual_backward(loss)
            if batch_idx % self.conf.trainer.accumulate_grad_batches == 0:
                self.clip_gradients(optimizer, 1)
                optimizer.step()
                optimizer.zero_grad()
                lr_sched.step()
            softmax = F.softmax(logits, dim=1)
            return {"loss": loss, "softmax": softmax, "labels": y, "confid": None}

        if self.training_stage == 1:
            lr_sched = self.lr_schedulers()[1]

            x, y = batch
            outputs = self.network(x)
            softmax = F.softmax(outputs[0], dim=1)
            pred_confid = torch.sigmoid(outputs[1])
            tcp = softmax.gather(1, y.unsqueeze(1))
            loss = (
                F.mse_loss(pred_confid, tcp) / self.conf.trainer.accumulate_grad_batches
            )
            self.manual_backward(loss)
            if batch_idx % self.conf.trainer.accumulate_grad_batches == 0:
                self.clip_gradients(optimizer, 1)
                optimizer.step()
                optimizer.zero_grad()
                if self.trainer.is_last_batch:
                    lr_sched.step()
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
            loss = (
                F.mse_loss(pred_confid, tcp) / self.conf.trainer.accumulate_grad_batches
            )
            self.manual_backward(loss)
            if batch_idx % self.conf.trainer.accumulate_grad_batches == 0:
                self.clip_gradients(optimizer, 1)
                optimizer.step()
                optimizer.zero_grad()
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
        # one optimizer per training stage
        optimizers = [
            # backbone training
            self.conf.trainer.optimizer(self.backbone.parameters()),
            # confidnet training
            torch.optim.Adam(
                self.network.confid_net.parameters(),
                lr=self.conf.trainer.learning_rate_confidnet,
            ),
            # backbone fine-tuning
            torch.optim.Adam(
                self.network.parameters(),
                lr=self.conf.trainer.learning_rate_confidnet_finetune,
            ),
        ]

        schedulers = [
            self.conf.trainer.lr_scheduler(optimizers[0]),
            {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizers[1],
                    T_max=self.milestones[1] - self.milestones[0],
                    verbose=True,
                ),
                "interval": "epoch",
                "frequency": 1,
                "name": "confidnet_adam",
            },
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

    def freeze_layers(self, model, freeze_string=None, keep_string=None):
        for param in model.named_parameters():
            if freeze_string is None and keep_string is None:
                param[1].requires_grad = False
            if freeze_string is not None and freeze_string in param[0]:
                param[1].requires_grad = False
            if keep_string is not None and keep_string not in param[0]:
                param[1].requires_grad = False

    def unfreeze_layers(self, model, unfreeze_string=None):
        for param in model.named_parameters():
            if unfreeze_string is None or unfreeze_string in param[0]:
                param[1].requires_grad = True

    def disable_bn(self, model):
        # Freeze also BN running average parameters
        for layer in model.named_modules():
            if (
                "bn" in layer[0]
                or "cbr_unit.1" in layer[0]
                or isinstance(layer[1], torch.nn.BatchNorm2d)
            ):
                layer[1].momentum = 0
                layer[1].eval()

    def disable_dropout(self, model):
        for layer in model.named_modules():
            if "dropout" in layer[0] or isinstance(layer[1], torch.nn.Dropout):
                layer[1].eval()

    def check_weight_consistency(self, pl_module):
        for ix, x in enumerate(pl_module.backbone.named_parameters()):
            if ix == 0:
                logger.debug("BACKBONE {} {}", x[0], x[1].mean().item())

        for ix, x in enumerate(pl_module.network.encoder.named_parameters()):
            if ix == 0:
                logger.debug("CONFID ENCODER {} {}", x[0], x[1].mean().item())

        for ix, x in enumerate(pl_module.network.confid_net.named_parameters()):
            if ix == 0:
                logger.debug("CONFIDNET {} {}", x[0], x[1].mean().item())

        for ix, x in enumerate(pl_module.network.classifier.named_parameters()):
            if ix == 0:
                logger.debug("CONFID CLassifier {} {}", x[0], x[1].mean().item())
