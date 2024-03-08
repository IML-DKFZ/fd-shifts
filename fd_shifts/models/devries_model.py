from __future__ import annotations

import re
from typing import TYPE_CHECKING

import hydra
import pl_bolts
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from fd_shifts import logger
from fd_shifts.models.networks import get_network
from fd_shifts.utils import to_dict

if TYPE_CHECKING:
    from fd_shifts import configs


class net(pl.LightningModule):
    """

    Attributes:
        optimizer_cfgs:
        lr_scheduler_cfgs:
        query_confids:
        num_epochs:
        num_classes:
        nll_loss:
        cross_entropy_loss:
        lmbda:
        budget:
        test_conf_scaling:
        ext_confid_name:
        imagenet_weights_path:
        model:
        test_mcd_samples:
        monitor_mcd_samples:
        test_results:
        loaded_epoch:
    """

    def __init__(self, cf: configs.Config):
        super(net, self).__init__()

        self.save_hyperparameters(to_dict(cf))

        self.cf = cf

        self.optimizer_cfgs = cf.trainer.optimizer
        self.lr_scheduler_cfgs = cf.trainer.lr_scheduler
        self.lr_scheduler_interval = cf.trainer.lr_scheduler_interval

        if cf.trainer.callbacks["model_checkpoint"] is not None:
            logger.info(
                "Initializing custom Model Selector. {}",
                cf.trainer.callbacks.model_checkpoint,
            )
            self.selection_metrics = cf.trainer.callbacks["model_checkpoint"][
                "selection_metric"
            ]
            self.selection_modes = cf.trainer.callbacks["model_checkpoint"]["mode"]

        self.query_confids = cf.eval.confidence_measures
        self.num_epochs = cf.trainer.num_epochs
        self.num_classes = cf.data.num_classes
        self.nll_loss = nn.NLLLoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.lmbda = 0.1
        self.budget = cf.model.budget
        self.test_conf_scaling = cf.eval.test_conf_scaling
        self.ext_confid_name = cf.eval.ext_confid_name
        self.imagenet_weights_path = cf.model.network.__dict__.get(
            "imagenet_weights_path"
        )

        if self.ext_confid_name == "dg":
            self.reward = cf.model.dg_reward
            self.pretrain_epochs = cf.trainer.dg_pretrain_epochs
            self.pretrain_steps = cf.trainer.dg_pretrain_steps
            self.load_dg_backbone_path = cf.model.network.__dict__.get(
                "load_dg_backbone_path"
            )
            self.save_dg_backbone_path = cf.model.network.__dict__.get(
                "save_dg_backbone_path"
            )

        self.model = get_network(cf.model.network.name)(cf)

        self.test_mcd_samples = cf.model.test_mcd_samples
        self.monitor_mcd_samples = cf.model.monitor_mcd_samples

    def forward(self, x):
        return self.model(x)

    def mcd_eval_forward(self, x, n_samples):
        self.model.encoder.enable_dropout()

        softmax_list = []
        conf_list = []
        for _ in range(n_samples - len(softmax_list)):
            if self.ext_confid_name == "devries":
                logits, confidence = self.model(x)
                softmax = F.softmax(logits, dim=1)
                confidence = torch.sigmoid(confidence).squeeze(1)
                softmax_list.append(logits.unsqueeze(2))
                conf_list.append(confidence.unsqueeze(1))
            if self.ext_confid_name == "dg":
                outputs = self.model(x)
                soutputs = F.softmax(outputs, dim=1)
                softmax, reservation = soutputs[:, :-1], soutputs[:, -1]
                confidence = 1 - reservation
                softmax_list.append(outputs[:, :-1].unsqueeze(2).detach())
                conf_list.append(confidence.unsqueeze(1).detach())

        self.model.encoder.disable_dropout()

        return torch.cat(softmax_list, dim=2), torch.cat(conf_list, dim=1)

    def on_epoch_end(self):
        if (
            self.ext_confid_name == "dg"
            and (
                (
                    self.pretrain_epochs is not None
                    and self.current_epoch == self.pretrain_epochs - 1
                )
                or (
                    self.pretrain_steps is not None
                    and self.global_step == self.pretrain_steps - 1
                )
            )
            and self.save_dg_backbone_path is not None
        ):
            self.trainer.save_checkpoint(self.save_dg_backbone_path)
            tqdm.write(
                "saved pretrained dg backbone to {}".format(self.save_dg_backbone_path)
            )

    def on_train_start(self):
        if self.imagenet_weights_path:
            self.model.encoder.load_pretrained_imagenet_params(
                self.imagenet_weights_path
            )

        if self.current_epoch > 0:
            tqdm.write("stepping scheduler after resume...")
            self.trainer.lr_schedulers[0]["scheduler"].step()

        for ix, x in enumerate(self.model.named_modules()):
            tqdm.write(str(ix))
            tqdm.write(str(x[1]))
            if isinstance(x[1], nn.Conv2d) or isinstance(x[1], nn.Linear):
                tqdm.write(str(x[1].weight.mean().item()))

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.ext_confid_name == "devries":
            logits, confidence = self.model(x)
            confidence = torch.sigmoid(confidence)
            pred_original = F.softmax(logits, dim=1)
            labels_onehot = torch.nn.functional.one_hot(y, num_classes=self.num_classes)
            # Make sure we don't have any numerical instability
            eps = 1e-12
            pred_original = torch.clamp(pred_original, 0.0 + eps, 1.0 - eps)
            confidence = torch.clamp(confidence, 0.0 + eps, 1.0 - eps)

            # Randomly set half of the confidences to 1 (i.e. no hints)
            b = torch.bernoulli(torch.Tensor(confidence.size()).uniform_(0, 1)).to(
                self.device
            )
            conf = confidence * b + (1 - b)
            pred_new = pred_original * conf.expand_as(pred_original) + labels_onehot * (
                1 - conf.expand_as(labels_onehot)
            )
            pred_new = torch.log(pred_new)

            xentropy_loss = self.nll_loss(pred_new, y)
            confidence_loss = torch.mean(-torch.log(confidence))

            loss = xentropy_loss + (self.lmbda * confidence_loss)
            if self.budget > confidence_loss:
                self.lmbda = self.lmbda / 1.01
            elif self.budget <= confidence_loss:
                self.lmbda = self.lmbda / 0.99

        elif self.ext_confid_name == "dg":
            logits = self.model(x)
            softmax = F.softmax(logits, dim=1)
            pred_original, reservation = softmax[:, :-1], softmax[:, -1]
            confidence = 1 - reservation.unsqueeze(1)
            if (
                (
                    self.pretrain_epochs is not None
                    and self.current_epoch >= self.pretrain_epochs
                )
                or (
                    self.pretrain_steps is not None
                    and self.global_step >= self.pretrain_steps
                )
            ) and self.reward > -1:
                gain = torch.gather(
                    pred_original, dim=1, index=y.unsqueeze(1)
                ).squeeze()
                doubling_rate = (gain.add(reservation.div(self.reward))).log()
                loss = -doubling_rate.mean().unsqueeze(0)
            else:
                loss = self.cross_entropy_loss(logits[:, :-1], y)

        return {
            "loss": loss,
            "softmax": pred_original,
            "labels": y,
            "confid": confidence.squeeze(1),
        }

    def training_step_end(self, batch_parts):
        batch_parts["loss"] = batch_parts["loss"].mean()
        return batch_parts

    def validation_step(self, batch, batch_idx):
        x, y = batch

        if self.ext_confid_name == "devries":
            logits, confidence = self.model(x)
            confidence = torch.sigmoid(confidence)
            pred_original = F.softmax(logits, dim=1)
            labels_onehot = torch.nn.functional.one_hot(y, num_classes=self.num_classes)

            # Make sure we don't have any numerical instability
            eps = 1e-12
            pred_original = torch.clamp(pred_original, 0.0 + eps, 1.0 - eps)
            confidence = torch.clamp(confidence, 0.0 + eps, 1.0 - eps)

            # Randomly set half of the confidences to 1 (i.e. no hints)
            b = torch.bernoulli(torch.Tensor(confidence.size()).uniform_(0, 1)).to(
                self.device
            )
            conf = confidence * b + (1 - b)
            pred_new = pred_original * conf.expand_as(pred_original) + labels_onehot * (
                1 - conf.expand_as(labels_onehot)
            )
            pred_new = torch.log(pred_new)

            xentropy_loss = self.nll_loss(pred_new, y)
            confidence_loss = torch.mean(-torch.log(confidence))

            loss = xentropy_loss + (self.lmbda * confidence_loss)

        elif self.ext_confid_name == "dg":
            outputs = self.model(x)
            outputs = F.softmax(outputs, dim=1)
            pred_original, reservation = outputs[:, :-1], outputs[:, -1]
            confidence = 1 - reservation.unsqueeze(1)
            if (
                (
                    self.pretrain_epochs is not None
                    and self.current_epoch >= self.pretrain_epochs
                )
                or (
                    self.pretrain_steps is not None
                    and self.global_step >= self.pretrain_steps
                )
            ) and self.reward > -1:
                gain = torch.gather(
                    pred_original, dim=1, index=y.unsqueeze(1)
                ).squeeze()
                doubling_rate = (gain.add(reservation.div(self.reward))).log()
                loss = -doubling_rate.mean()
            else:
                loss = self.cross_entropy_loss(outputs[:, :-1], y)

        return {
            "loss": loss,
            "softmax": pred_original,
            "labels": y,
            "confid": confidence.squeeze(1),
        }

    def validation_step_end(self, batch_parts):
        return batch_parts

    def test_step(self, batch, batch_idx, dataloader_idx, *args):
        x, y = batch
        z = self.model.forward_features(x)
        if self.ext_confid_name == "devries":
            logits, confidence = self.model.head(z)
            confidence = torch.sigmoid(confidence).squeeze(1)
        elif self.ext_confid_name == "dg":
            outputs = self.model.head(z)
            logits = outputs[:, :-1]
            soutputs = F.softmax(outputs, dim=1)
            softmax, reservation = soutputs[:, :-1], soutputs[:, -1]
            confidence = 1 - reservation
        else:
            raise NotImplementedError

        logits_dist = None
        confid_dist = None
        if any("mcd" in cfd for cfd in self.query_confids.test) and (
            not (self.cf.test.compute_train_encodings and dataloader_idx == 0)
        ):
            logits_dist, confid_dist = self.mcd_eval_forward(
                x=x, n_samples=self.test_mcd_samples
            )

        return {
            "logits": logits,
            "labels": y,
            "confid": confidence,
            "logits_dist": logits_dist,
            "confid_dist": confid_dist,
            "encoded": z,
        }

    def configure_optimizers(self):
        # optimizers = [
        #     hydra.utils.instantiate(self.optimizer_cfgs, _partial_=True)(
        #         self.model.parameters()
        #     )
        # ]

        # schedulers = [
        #     {
        #         "scheduler": hydra.utils.instantiate(self.lr_scheduler_cfgs)(
        #             optimizer=optimizers[0]
        #         ),
        #         "interval": self.lr_scheduler_interval,
        #     },
        # ]

        optimizers = [
            self.optimizer_cfgs(self.model.parameters()),
        ]

        schedulers = [
            {
                "scheduler": self.lr_scheduler_cfgs(optimizers[0]),
                "interval": self.lr_scheduler_interval,
            },
        ]

        return optimizers, schedulers

    def on_load_checkpoint(self, checkpoint):
        self.loaded_epoch = checkpoint["epoch"]
        logger.info("loading checkpoint from epoch {}".format(self.loaded_epoch))

    def load_only_state_dict(self, path: str | Path) -> None:
        ckpt = torch.load(path)

        pattern = re.compile(r"^(\w*\.)(encoder|classifier)(\..*)")

        # For backwards-compatibility with before commit 1bdc717
        for param in list(ckpt["state_dict"].keys()):
            if param.startswith("model.classifier.module.model.features"):
                del ckpt["state_dict"][param]
                continue
            if param.startswith("model.classifier.module.model.classifier"):
                correct_param = param.replace(".model.classifier", "")
                ckpt["state_dict"][correct_param] = ckpt["state_dict"][param]
                del ckpt["state_dict"][param]
                param = correct_param
            if pattern.match(param):
                correct_param = re.sub(pattern, r"\1_\2\3", param)
                ckpt["state_dict"][correct_param] = ckpt["state_dict"][param]
                del ckpt["state_dict"][param]

        logger.info("loading checkpoint from epoch {}".format(ckpt["epoch"]))
        self.load_state_dict(ckpt["state_dict"], strict=True)

    def last_layer(self):
        state = self.state_dict()
        model_prefix = "model"
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
