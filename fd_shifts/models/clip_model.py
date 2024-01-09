from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import open_clip as oc
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torchvision import transforms

from fd_shifts import logger
from fd_shifts.utils import to_dict

if TYPE_CHECKING:
    from fd_shifts import configs


class ClipOodModel(pl.LightningModule):
    def __init__(self, cfg: configs.Config):
        super().__init__()
        self.save_hyperparameters(to_dict(cfg))
        self.conf = cfg

        self.class_prefix = None
        self.model, _, self.preprocess = oc.create_model_and_transforms(
            "ViT-B-16",
            pretrained="laion2b_s34b_b88k",
        )
        self.tokenizer = oc.get_tokenizer("ViT-B-16")

        # example for directly logging metrics
        # self.accuracy = Accuracy(
        #     task="multiclass",
        #     num_classes=self.conf.data.num_classes,
        #     validate_args=False,
        # )

    def on_test_start(self):
        self.datasets = list(
            map(lambda d: d.dataset, self.trainer.datamodule.test_dataloader())
        )

        if hasattr(self.datasets[0], "classes"):
            classes = self.datasets[0].classes
        else:
            classes = list(map(str, range(self.conf.data.num_classes)))

        if self.class_prefix is not None:
            classes = list(map(lambda c: f"{self.class_prefix} {c}", classes))

        logger.debug(f"{classes=}")

        text = self.tokenizer(classes).to(self.device)
        self.text_features = self.model.encode_text(text)
        self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    def test_step(self, batch, batch_idx, dataset_idx):
        x, y = batch
        x = transforms.functional.resize(x, 224)

        image_features = self.model.encode_image(x)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        logits = image_features @ self.text_features.T

        # example for directly logging metrics
        # text_probs = torch.softmax(logits, dim=-1)
        # preds = torch.argmax(text_probs, dim=-1)
        # self.accuracy(preds, y)
        # self.log(
        #     f"accuracy",
        #     self.accuracy,
        # )

        # only set this if you want to write this to disk, complains if test_results exists and is anything but a dict with these keys
        self.test_results = {
            "logits": logits,
            "logits_dist": None,
            "labels": y,
            "confid": None,
            "confid_dist": None,
            "encoded": None,
        }

    def load_only_state_dict(self, path: str | Path) -> None:
        pass
