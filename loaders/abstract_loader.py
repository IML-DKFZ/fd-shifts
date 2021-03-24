from pathlib import Path
import numpy as np
import torch
import os
from torch.utils.data.sampler import SubsetRandomSampler
import pytorch_lightning as pl

from imlone.utils.aug_utils import transforms_collection
from confidnet.augmentations import get_composed_augmentations

class AbstractDataLoader():

    def __init__(self, cf):

        super().__init__()
        self.exp_dir = Path(cf.exp.dir)
  #      self.resume_dir = cf.exp_dir # ???
        self.data_dir = os.path.join(os.environ["DATASET_ROOT_DIR"], cf.data.dataset)
        self.batch_size = cf.trainer.batch_size
        self.img_size = cf.data.img_size
       # self.resume_dir = config_args['model']['resume'].parent if isinstance(config_args['model']['resume'], Path) else None
        self.val_ratio = cf.trainer.val_ratio
     #   self.perturbed_folder = config_args['data'].get('perturbed_images', None)
        self.pin_memory = cf.data.pin_memory
        self.num_workers = cf.data.num_workers

        # Set up augmentations
        self.augmentations = {}
        if cf.augmentations:
            self.add_augmentations(cf.augmentations)
        else:
            self.augmentations["train"] = transforms_collection["to_tensor"]()
            self.augmentations["val"] = transforms_collection["to_tensor"]()
            self.augmentations["test"] = None
            print("NO AUGMENTATIONS!!!!")

        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None

        self.prepare_data()
        self.setup()
        self.make_train_dataloader()
        self.make_val_dataloader()


    def add_augmentations(self, query_augs):

        self.augmentations["train"] = get_composed_augmentations(query_augs["train"], verbose=False, training="classif")
        # for datasplit_k, datasplit_v in query_augs.items():
        #     augmentations, aug_after = [], []
        #     for aug_key, aug_param in datasplit_v.items():
        #         if aug_key == "normalize":
        #             aug_after.append(transforms_collection[aug_key](aug_param))
        #             continue
        #         augmentations.append(transforms_collection[aug_key](aug_param))
        #
        #     augmentations.append(transforms_collection["to_tensor"]())
        #     # augmentations.extend(aug_after)
        #     print("AUGMENTATIONS", augmentations)
        #     self.augmentations[datasplit_k] = transforms_collection["compose"](augmentations)


    def prepare_data(self):
        pass


    def setup(self, stage=None):

        # self.prepare_dataset()
        num_train = len(self.train_dataset)
        indices = list(range(num_train))

        if (self.exp_dir / "train_idx.npy").exists():
            train_idx = np.load(self.exp_dir / "train_idx.npy")
            val_idx = np.load(self.exp_dir / "val_idx.npy")

        # Splitting indices
        # elif self.resume_dir:
        #     LOGGER.warning("Loading existing train-val split indices from ORIGINAL training")
        #     train_idx = np.load(self.resume_dir / "train_idx.npy")
        #     val_idx = np.load(self.resume_dir / "val_idx.npy")
        else:
            split = int(np.floor(self.val_ratio * num_train))
            np.random.shuffle(indices)
            train_idx, val_idx = indices[split:], indices[:split]
            np.save(self.exp_dir / "train_idx.npy", train_idx)
            np.save(self.exp_dir / "val_idx.npy", val_idx)

        # Make samplers
        self.train_sampler = SubsetRandomSampler(train_idx)
        self.val_sampler = SubsetRandomSampler(val_idx)


    def make_train_dataloader(self):
        self.train_dataloader = torch.utils.data.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            )


    def make_val_dataloader(self):
        self.val_dataloader = torch.utils.data.DataLoader(
            dataset=self.train_dataset, #same dataset as train but potentially differing augs.
            batch_size=self.batch_size,
            sampler=self.val_sampler,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            )


    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
        )


